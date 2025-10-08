#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include "pacmap_utils.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <unordered_set>
#include <memory>
#include <iostream>


void sample_triplets(PacMapModel* model, float* data, pacmap_progress_callback_internal callback) {

    // CRITICAL DEBUG: Add more detailed triplet analysis

    // Validate parameters first
    if (validate_parameters(model) != PACMAP_SUCCESS) {
        if (callback) callback("Error", 0, 100, 0.0f, "Parameter validation failed");
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

    if (!model->original_space_index) {
        if (callback) callback("Error", 0, 100, 0.0f, "Failed to create HNSW index");
        return;
    }

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

              
    // CRITICAL DEBUG: Analyze triplet quality
    if (model->triplets.size() > 0) {
        // Check if triplets have good diversity in indices
        std::unordered_set<int> unique_anchors, unique_neighbors;
        std::unordered_map<TripletType, int> type_counts;

        for (const auto& triplet : model->triplets) {
            unique_anchors.insert(triplet.anchor);
            unique_neighbors.insert(triplet.neighbor);
            type_counts[triplet.type]++;
        }

        printf("[PAIR DEBUG] Triplet Analysis:\n");
        printf("  - Total triplets: %zu\n", model->triplets.size());
        printf("  - Unique anchors: %zu (%.1f%% of all points)\n",
               unique_anchors.size(), 100.0f * unique_anchors.size() / model->n_samples);
        printf("  - Unique neighbors: %zu\n", unique_neighbors.size());
        printf("  - Type distribution: NEIGHBOR=%d, MID_NEAR=%d, FURTHER=%d\n",
               type_counts[NEIGHBOR], type_counts[MID_NEAR], type_counts[FURTHER]);

        // Sample first few triplets for inspection with distance analysis
        printf("[PAIR DEBUG] Sample triplets (first 10):\n");
        for (int i = 0; i < std::min(10, (int)model->triplets.size()); ++i) {
            const auto& t = model->triplets[i];
            float distance = compute_sampling_distance(
                normalized_data.data() + t.anchor * model->n_features,
                normalized_data.data() + t.neighbor * model->n_features,
                model->n_features, model->metric);

            const char* type_str = (t.type == NEIGHBOR) ? "NEIGHBOR" :
                                   (t.type == MID_NEAR) ? "MID_NEAR" : "FURTHER";
            printf("  %d: (%d,%d) type=%s dist=%.3f\n", i, t.anchor, t.neighbor, type_str, distance);
        }

        // Distance statistics by type
        printf("[PAIR DEBUG] Distance statistics by triplet type:\n");
        for (auto const& [type, count] : type_counts) {
            if (count == 0) continue;

            float min_dist = std::numeric_limits<float>::infinity();
            float max_dist = 0.0f;
            float total_dist = 0.0f;
            int valid_samples = 0;

            for (const auto& t : model->triplets) {
                if (t.type != type) continue;

                float dist = compute_sampling_distance(
                    normalized_data.data() + t.anchor * model->n_features,
                    normalized_data.data() + t.neighbor * model->n_features,
                    model->n_features, model->metric);

                min_dist = std::min(min_dist, dist);
                max_dist = std::max(max_dist, dist);
                total_dist += dist;
                valid_samples++;
            }

            const char* type_str = (type == NEIGHBOR) ? "NEIGHBOR" :
                                   (type == MID_NEAR) ? "MID_NEAR" : "FURTHER";
            printf("  %s: count=%d, dist_range=[%.3f, %.3f], avg=%.3f\n",
                   type_str, count, min_dist, max_dist,
                   valid_samples > 0 ? total_dist / valid_samples : 0.0f);
        }
    }

    // Update triplet counts
    model->total_triplets = static_cast<int>(model->triplets.size());
    model->neighbor_triplets = static_cast<int>(neighbor_triplets.size());
    model->mid_near_triplets = static_cast<int>(mn_triplets.size());
    model->far_triplets = static_cast<int>(fp_triplets.size());

    printf("[TRIPLET DEBUG] Completed sampling: %zu total triplets generated\n", model->triplets.size());
    callback("Sampling Triplets", 100, 100, 100.0f, nullptr);
}

void sample_neighbors_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                         std::vector<Triplet>& neighbor_triplets) {

    neighbor_triplets.clear();
    neighbor_triplets.reserve(model->n_samples * model->n_neighbors);
    printf("[DEBUG] Using PYTHON-STYLE neighbor pair sampling (simple sklearn approach)\n");

    // PYTHON-style approach: Exactly like sklearn NearestNeighbors
    // Find k+1 neighbors (including self), then skip self when creating pairs

    if (model->force_exact_knn) {
        printf("[DEBUG] Using EXACT k-NN (brute-force) like Python sklearn\n");

        #pragma omp parallel for if(model->n_samples > 1000)
        for (int i = 0; i < model->n_samples; ++i) {
            std::vector<std::pair<float, int>> knn;
            distance_metrics::find_knn_exact(
                normalized_data.data() + i * model->n_features,
                normalized_data.data(),
                model->n_samples,
                model->n_features,
                model->metric,
                model->n_neighbors + 1,  // k_neighbors + 1 (includes self, like Python)
                knn,
                i  // query_index to skip self
            );

            // Python style: skip first neighbor (self) and use the rest
            // Start from j=1 to skip self, just like Python's indices[i, 1:]
            #pragma omp critical
            {
                for (int j = 1; j < model->n_neighbors + 1 && j < static_cast<int>(knn.size()); ++j) {
                    int neighbor_idx = knn[j].second;
                    neighbor_triplets.emplace_back(i, neighbor_idx, NEIGHBOR);
                }
            }
        }
    } else {
        printf("[DEBUG] Using HNSW k-NN like Python sklearn\n");

        #pragma omp parallel for if(model->n_samples > 1000)
        for (int i = 0; i < model->n_samples; ++i) {
            auto knn_results = model->original_space_index->searchKnn(
                normalized_data.data() + i * model->n_features,
                model->n_neighbors + 1);  // k_neighbors + 1 (includes self, like Python)

            // Convert priority_queue to vector and reverse to get sorted order
            std::vector<std::pair<float, size_t>> knn;
            while (!knn_results.empty()) {
                knn.push_back(knn_results.top());
                knn_results.pop();
            }
            std::reverse(knn.begin(), knn.end());

            // Python style: skip first neighbor (self) and use the rest
            // Start from j=1 to skip self, just like Python's indices[i, 1:]
            #pragma omp critical
            {
                for (int j = 1; j < model->n_neighbors + 1 && j < static_cast<int>(knn.size()); ++j) {
                    int neighbor_idx = static_cast<int>(knn[j].second);
                    neighbor_triplets.emplace_back(i, neighbor_idx, NEIGHBOR);
                }
            }
        }
    }

    printf("[DEBUG] PYTHON-STYLE neighbor sampling completed: %d neighbor triplets generated\n", (int)neighbor_triplets.size());
}

void sample_MN_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& mn_triplets, int n_mn) {

    mn_triplets.clear();
    printf("[DEBUG] CRITICAL FIX: Using PER-POINT RANDOM MN sampling (matching Rust implementation)\n");

    // CRITICAL FIX: Switch to per-point random sampling (matching Rust sampling.rs)
    // For each point i, sample 6 random points, compute distances, sort, pick 2nd closest
    mn_triplets.reserve(model->n_samples * n_mn);
    std::unordered_set<long long> used_pairs;

    #pragma omp parallel
    {
        std::mt19937 rng = get_seeded_rng(model->random_seed + omp_get_thread_num());
        std::uniform_int_distribution<int> dist(0, model->n_samples - 1);
        std::vector<std::pair<float, int>> distances;
        distances.reserve(6); // Sample 6 candidates as in Rust

        #pragma omp for
        for (int i = 0; i < model->n_samples; ++i) {
            distances.clear();
            std::unordered_set<int> sampled_indices;

            // Sample 6 unique random points
            while (sampled_indices.size() < 6) {
                int j = dist(rng);
                if (j != i && sampled_indices.find(j) == sampled_indices.end()) {
                    float d = compute_sampling_distance(
                        normalized_data.data() + i * model->n_features,
                        normalized_data.data() + j * model->n_features,
                        model->n_features, PACMAP_METRIC_EUCLIDEAN);  // CRITICAL FIX: Always use Euclidean for MN pairs
                    distances.emplace_back(d, j);
                    sampled_indices.insert(j);
                }
            }

            // Sort by distance and pick second closest (index 1)
            std::sort(distances.begin(), distances.end());
            for (int k = 0; k < n_mn && k + 1 < distances.size(); ++k) {
                int j = distances[k + 1].second; // Skip closest (index 0), pick 2nd, 3rd, etc.
                long long pair_key = ((long long)std::min(i, j) << 32) | std::max(i, j);
                #pragma omp critical
                {
                    if (used_pairs.find(pair_key) == used_pairs.end()) {
                        mn_triplets.emplace_back(Triplet{i, j, MID_NEAR});
                        used_pairs.insert(pair_key);
                    }
                }
            }
        }
    }

    printf("[DEBUG] PER-POINT MN sampling completed: %d triplets generated\n", (int)mn_triplets.size());
}

void sample_FP_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& fp_triplets, int n_fp) {

    fp_triplets.clear();
    printf("[DEBUG] CRITICAL FIX: Using RANDOM FP sampling without distance filtering (matching Rust implementation)\n");

    // CRITICAL FIX: Switch to pure random sampling (matching Rust sampling.rs)
    // Sample random points excluding neighbors and self, no distance filtering
    fp_triplets.reserve(model->n_samples * n_fp);
    std::unordered_set<long long> used_pairs;

    // Get neighbor indices for exclusion
    std::vector<std::unordered_set<int>> neighbor_sets(model->n_samples);
    for (const auto& t : model->triplets) {
        if (t.type == NEIGHBOR) {
            neighbor_sets[t.anchor].insert(t.neighbor);
        }
    }

    #pragma omp parallel
    {
        std::mt19937 rng = get_seeded_rng(model->random_seed + 1000 + omp_get_thread_num());
        std::uniform_int_distribution<int> dist(0, model->n_samples - 1);

        #pragma omp for
        for (int i = 0; i < model->n_samples; ++i) {
            std::unordered_set<int> sampled_indices;
            int found = 0;

            while (found < n_fp) {
                int j = dist(rng);
                if (j != i &&
                    sampled_indices.find(j) == sampled_indices.end() &&
                    neighbor_sets[i].find(j) == neighbor_sets[i].end()) {

                    long long pair_key = ((long long)std::min(i, j) << 32) | std::max(i, j);
                    #pragma omp critical
                    {
                        if (used_pairs.find(pair_key) == used_pairs.end()) {
                            fp_triplets.emplace_back(Triplet{i, j, FURTHER});
                            used_pairs.insert(pair_key);
                            found++;
                        }
                    }
                    sampled_indices.insert(j);
                }
            }
        }
    }

    printf("[DEBUG] RANDOM FP sampling completed: %d triplets generated\n", (int)fp_triplets.size());
}

void distance_based_sampling(PacMapModel* model, const std::vector<float>& data,
                           int oversample_factor, int target_pairs, float min_dist, float max_dist,
                           std::vector<Triplet>& triplets, TripletType type) {

    std::uniform_int_distribution<int> dist(0, model->n_samples - 1);
    std::unordered_set<long long> used_pairs;
    int pairs_found = 0;

    // Adaptive sampling loop with oversampling
    int max_attempts = oversample_factor;  // Use oversample_factor parameter
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

std::vector<float> compute_distance_percentiles(const std::vector<float>& data, int n_samples, int n_features, PacMapMetric metric) {
    std::vector<float> distances;

    // Sample distances for percentile estimation (optimize for large datasets)
    int sample_size = std::min(n_samples, 1000);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = i + 1; j < sample_size; ++j) {
            float dist = compute_sampling_distance(data.data() + i * n_features,
                                                 data.data() + j * n_features,
                                                 n_features, metric);  // CRITICAL FIX: Use actual metric!
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
    return distance_metrics::compute_distance(x, y, n_features, metric);
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
}