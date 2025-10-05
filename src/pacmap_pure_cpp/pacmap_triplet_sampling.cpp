#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include "pacmap_utils.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <unordered_set>
#include <memory>
#include <iostream>

// Utility function for RNG management
std::mt19937 get_seeded_rng(int seed) {
    return seed >= 0 ? std::mt19937(seed) : std::mt19937(std::random_device{}());
}

void sample_triplets(PacMapModel* model, float* data, uwot_progress_callback_v2 callback) {
    // Validate parameters first
    if (validate_parameters(model) != PACMAP_SUCCESS) {
        return;
    }

    // Normalize/quantize data (reuse UMAP infrastructure)
    std::vector<float> normalized_data;
    normalized_data.assign(data, data + model->n_samples * model->n_features);

    if (model->use_quantization) {
        // Apply quantization (reuse from existing infrastructure)
        // quantize_data(normalized_data, model->n_features);
    }

    // Save normalized data for transform
    model->training_data = normalized_data;

    // Build HNSW index for efficient neighbor search
    model->original_space_index = create_hnsw_index(normalized_data.data(),
                                                    model->n_samples,
                                                    model->n_features,
                                                    model->metric,
                                                    model->hnsw_m,
                                                    model->hnsw_ef_construction);

    // Initialize RNG for deterministic behavior
    model->rng = get_seeded_rng(model->random_seed);

    // Sample three types of triplets
    std::vector<Triplet> neighbor_triplets, mn_triplets, fp_triplets;

    // Neighbor pairs using HNSW
    sample_neighbors_pair(model, normalized_data, neighbor_triplets);

    // Mid-near pairs with distance-based sampling
    int n_mn = static_cast<int>(model->n_neighbors * model->mn_ratio);
    sample_MN_pair(model, normalized_data, mn_triplets, n_mn);

    // Far pairs for uniform distribution
    int n_fp = static_cast<int>(model->n_neighbors * model->fp_ratio);
    sample_FP_pair(model, normalized_data, fp_triplets, n_fp);

    // Combine all triplets (review-optimized: single vector)
    model->triplets.clear();
    model->triplets.reserve(neighbor_triplets.size() + mn_triplets.size() + fp_triplets.size());
    model->triplets.insert(model->triplets.end(), neighbor_triplets.begin(), neighbor_triplets.end());
    model->triplets.insert(model->triplets.end(), mn_triplets.begin(), mn_triplets.end());
    model->triplets.insert(model->triplets.end(), fp_triplets.begin(), fp_triplets.end());

    callback("Sampling Triplets", 100, 100, 100.0f, nullptr);
}

void sample_neighbors_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                         std::vector<Triplet>& neighbor_triplets) {

    neighbor_triplets.clear();
    neighbor_triplets.reserve(model->n_samples * model->n_neighbors);

    // Parallel neighbor sampling using HNSW (review optimization)
    #pragma omp parallel for
    for (int i = 0; i < model->n_samples; ++i) {
        auto knn = model->original_space_index->searchKnn(
            normalized_data.data() + i * model->n_features,
            model->n_neighbors + 1);

        std::vector<Triplet> local_triplets;
        for (size_t j = 1; j < knn.size(); ++j) {  // Skip self (first result)
            local_triplets.emplace_back(Triplet{i, static_cast<int>(knn[j].second), NEIGHBOR});
        }

        // Merge results safely
        #pragma omp critical
        {
            neighbor_triplets.insert(neighbor_triplets.end(),
                                   local_triplets.begin(), local_triplets.end());
        }
    }
}

void sample_MN_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& mn_triplets, int n_mn) {

    mn_triplets.clear();

    // Compute distance percentiles for mid-near range (25th-75th percentile)
    auto percentiles = compute_distance_percentiles(normalized_data,
                                                   std::min(model->n_samples, 1000),
                                                   model->n_features);
    float p25_dist = percentiles[0];
    float p75_dist = percentiles[1];

    // Distance-based sampling for mid-near pairs
    distance_based_sampling(model, normalized_data,
                           model->n_samples * n_mn * 2,  // Oversample for uniqueness
                           p25_dist, p75_dist,
                           mn_triplets, MID_NEAR);
}

void sample_FP_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& fp_triplets, int n_fp) {

    fp_triplets.clear();

    // Compute 90th percentile for far pairs
    auto percentiles = compute_distance_percentiles(normalized_data,
                                                   std::min(model->n_samples, 500),
                                                   model->n_features);
    float p90_dist = percentiles[2];  // 90th percentile

    // Distance-based sampling for far pairs
    distance_based_sampling(model, normalized_data,
                           model->n_samples * n_fp * 3,  // Oversample more for far pairs
                           p90_dist, std::numeric_limits<float>::infinity(),
                           fp_triplets, FURTHER);
}

void distance_based_sampling(PacMapModel* model, const std::vector<float>& data,
                           int target_pairs, float min_dist, float max_dist,
                           std::vector<Triplet>& triplets, TripletType type) {

    std::uniform_int_distribution<int> dist(0, model->n_samples - 1);
    std::unordered_set<long long> used_pairs;
    int pairs_found = 0;

    // Adaptive sampling loop with oversampling
    int max_attempts = target_pairs * 10;  // Prevent infinite loops
    int attempts = 0;

    while (pairs_found < target_pairs && attempts < max_attempts) {
        int i = dist(model->rng);
        int j = dist(model->rng);

        if (i == j) continue;

        // Ensure unique pairs using bit packing
        long long pair_key = ((long long)std::min(i, j) << 32) | std::max(i, j);
        if (used_pairs.find(pair_key) != used_pairs.end()) continue;

        // Compute distance with early termination for efficiency
        float distance = compute_sampling_distance(data.data() + i * model->n_features,
                                                 data.data() + j * model->n_features,
                                                 model->n_features, model->metric);

        if (distance >= min_dist && distance <= max_dist) {
            triplets.emplace_back(Triplet{i, j, type});
            used_pairs.insert(pair_key);
            pairs_found++;
        }

        attempts++;
    }
}

std::vector<float> compute_distance_percentiles(const std::vector<float>& data, int n_samples, int n_features) {
    std::vector<float> distances;

    // Sample distances for percentile estimation (optimize for large datasets)
    int sample_size = std::min(n_samples, 1000);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = i + 1; j < sample_size; ++j) {
            float dist = compute_sampling_distance(data.data() + i * n_features,
                                                 data.data() + j * n_features,
                                                 n_features, PACMAP_METRIC_EUCLIDEAN);
            distances.push_back(dist);
        }
    }

    std::sort(distances.begin(), distances.end());
    std::vector<float> percentiles(3);

    // 25th, 75th, and 90th percentiles
    percentiles[0] = distances[distances.size() * 0.25];
    percentiles[1] = distances[distances.size() * 0.75];
    percentiles[2] = distances[distances.size() * 0.90];

    return percentiles;
}

std::unique_ptr<hnswlib::HierarchicalNSW<float>> create_hnsw_index(
    const float* data, int n_samples, int n_features, PacMapMetric metric,
    int M, int ef_construction) {

    // Create appropriate space based on metric
    hnswlib::SpaceInterface<float>* space;
    switch (metric) {
        case PACMAP_METRIC_EUCLIDEAN:
            space = new hnswlib::L2Space(n_features);
            break;
        case PACMAP_METRIC_COSINE:
            space = new hnswlib::InnerProductSpace(n_features);
            break;
        default:
            space = new hnswlib::L2Space(n_features);
            break;
    }

    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space, n_samples, M, ef_construction);

    // Add data points to index
    for (int i = 0; i < n_samples; ++i) {
        index->addPoint(data + i * n_features, i);
    }

    return index;
}

bool is_valid_triplet(const Triplet& triplet, int n_samples) {
    return triplet.anchor >= 0 && triplet.anchor < n_samples &&
           triplet.neighbor >= 0 && triplet.neighbor < n_samples &&
           triplet.anchor != triplet.neighbor;
}

void filter_invalid_triplets(std::vector<Triplet>& triplets, int n_samples) {
    triplets.erase(
        std::remove_if(triplets.begin(), triplets.end(),
                      [n_samples](const Triplet& t) {
                          return !is_valid_triplet(t, n_samples);
                      }),
        triplets.end()
    );
}

void deduplicate_triplets(std::vector<Triplet>& triplets) {
    std::unordered_set<long long> seen;
    triplets.erase(
        std::remove_if(triplets.begin(), triplets.end(),
                      [&seen](const Triplet& t) {
                          long long key = ((long long)std::min(t.anchor, t.neighbor) << 32) |
                                         std::max(t.anchor, t.neighbor);
                          if (seen.find(key) != seen.end()) {
                              return true;
                          }
                          seen.insert(key);
                          return false;
                      }),
        triplets.end()
    );
}

float compute_sampling_distance(const float* x, const float* y, int n_features, PacMapMetric metric) {
    switch (metric) {
        case PACMAP_METRIC_EUCLIDEAN:
            return euclid_dist(x, y, n_features);
        case PACMAP_METRIC_COSINE:
            return angular_dist(x, y, n_features);
        case PACMAP_METRIC_MANHATTAN:
            return manhattan_dist(x, y, n_features);
        default:
            return euclid_dist(x, y, n_features);
    }
}

void print_sampling_statistics(const std::vector<Triplet>& triplets) {
    int neighbor_count = 0, mn_count = 0, fp_count = 0;

    for (const auto& t : triplets) {
        switch (t.type) {
            case NEIGHBOR: neighbor_count++; break;
            case MID_NEAR: mn_count++; break;
            case FURTHER: fp_count++; break;
        }
    }

    std::cout << "Triplet Sampling Statistics:" << std::endl;
    std::cout << "  Total triplets: " << triplets.size() << std::endl;
    std::cout << "  Neighbor pairs: " << neighbor_count << std::endl;
    std::cout << "  Mid-near pairs: " << mn_count << std::endl;
    std::cout << "  Far pairs: " << fp_count << std::endl;
}