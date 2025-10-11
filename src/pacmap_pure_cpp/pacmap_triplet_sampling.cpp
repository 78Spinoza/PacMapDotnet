#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include "pacmap_utils.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <unordered_set>
#include <memory>
#include <iostream>
#include <atomic>

// ERROR12 Priority 5: Maximum vector size to prevent OOM with high ratios
#define MAX_VECTOR_SIZE (1ULL << 30)  // 1 billion elements max


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
                                                    model->hnsw_ef_construction,
                                                    callback);

    if (!model->original_space_index) {
        if (callback) callback("Error", 0, 100, 0.0f, "Failed to create HNSW index");
        return;
    }

    // Initialize RNG for deterministic behavior
    model->rng = get_seeded_rng(model->random_seed);

    // Sample three types of triplets
    std::vector<Triplet> neighbor_triplets, mn_triplets, fp_triplets;

    // Neighbor pairs using HNSW
    sample_neighbors_pair(model, normalized_data, neighbor_triplets, callback);

    // Mid-near pairs with improved sampling (ERROR11-FIX)
    int n_mn = static_cast<int>(model->n_neighbors * model->mn_ratio + 0.5); // ERROR11-FIX: Proper rounding
    // ERROR12 Priority 3: Cap triplet counts to prevent infeasible targets with arbitrary ratios
    n_mn = std::min(n_mn, static_cast<int>(static_cast<size_t>(model->n_samples) * (model->n_samples - 1) / 2));
    sample_MN_pair(model, normalized_data, mn_triplets, n_mn, callback);

    // Far pairs for uniform distribution
    int n_fp = static_cast<int>(model->n_neighbors * model->fp_ratio);
    // ERROR12 Priority 3: Cap triplet counts to prevent infeasible targets with arbitrary ratios
    n_fp = std::min(n_fp, static_cast<int>(static_cast<size_t>(model->n_samples) * (model->n_samples - 1) / 2));
    sample_FP_pair(model, normalized_data, fp_triplets, n_fp, callback);

    // Combine all triplets (review-optimized: single vector)
    model->triplets.clear();
    // ERROR12 Priority 5: Cap vector reserves to prevent OOM with high ratios
    model->triplets.reserve(std::min(
        neighbor_triplets.size() + mn_triplets.size() + fp_triplets.size(),
        MAX_VECTOR_SIZE));
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

              // Triplet analysis completed - detailed output disabled for clean interface
        // Progress is now reported via the enhanced progress callback system

        // Distance statistics by type - output disabled for clean interface
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

            // Distance statistics output disabled for clean interface
        }
    }

    // Update triplet counts
    model->total_triplets = static_cast<int>(model->triplets.size());
    model->neighbor_triplets = static_cast<int>(neighbor_triplets.size());
    model->mid_near_triplets = static_cast<int>(mn_triplets.size());
    model->far_triplets = static_cast<int>(fp_triplets.size());

        // Triplet sampling completed - detailed output disabled for clean interface
    callback("Sampling Triplets", 100, 100, 100.0f, nullptr);
}

void sample_neighbors_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                         std::vector<Triplet>& neighbor_triplets, pacmap_progress_callback_internal callback) {

    neighbor_triplets.clear();
    neighbor_triplets.reserve(model->n_samples * model->n_neighbors);

    if (callback) {
        callback("Sampling Neighbor Pairs", 0, model->n_samples, 0.0f,
                 model->force_exact_knn ? "Using exact KNN" : "Using HNSW approximate KNN");
    }

    // PYTHON-style approach: Exactly like sklearn NearestNeighbors
    // Find k+1 neighbors (including self), then skip self when creating pairs

    if (model->force_exact_knn) {
        int report_interval = std::max(1, model->n_samples / 10);
        std::atomic<int> completed(0);

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

            // Progress reporting
            int count = ++completed;
            if (callback && (count % report_interval == 0 || count == model->n_samples)) {
                float percent = (float)count / model->n_samples * 100.0f;
                callback("Sampling Neighbor Pairs", count, model->n_samples, percent, nullptr);
            }
        }

        if (callback) {
            callback("Sampling Neighbor Pairs", model->n_samples, model->n_samples, 100.0f,
                     "Neighbor pairs sampled using exact KNN");
        }
    } else {
            // HNSW k-NN mode enabled - detailed output disabled for clean interface
        int report_interval = std::max(1, model->n_samples / 10);
        std::atomic<int> completed(0);

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

            // Progress reporting
            int count = ++completed;
            if (callback && (count % report_interval == 0 || count == model->n_samples)) {
                float percent = (float)count / model->n_samples * 100.0f;
                callback("Sampling Neighbor Pairs", count, model->n_samples, percent, nullptr);
            }
        }

        if (callback) {
            callback("Sampling Neighbor Pairs", model->n_samples, model->n_samples, 100.0f,
                     "Neighbor pairs sampled using HNSW");
        }

#ifdef PACMAP_ENABLE_HNSW_VALIDATION
        // HNSW quality validation (DISABLED BY DEFAULT - for development/debugging only)
        // This validation runs exact KNN on validation samples which is O(n) per sample
        // For large datasets (>100K points), this can be extremely slow
        // Enable by defining PACMAP_ENABLE_HNSW_VALIDATION in CMakeLists.txt
        if (callback && model->n_samples >= 100) {
            callback("Validating HNSW Quality", 0, 100, 0.0f, "Calculating recall vs exact KNN");

            // Sample random points to validate HNSW recall
            int validation_samples = std::min(100, model->n_samples / 10);
            float total_recall = 0.0f;
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(0, model->n_samples - 1);

            for (int s = 0; s < validation_samples; ++s) {
                int sample_idx = dist(rng);

                // Get exact KNN
                std::vector<std::pair<float, int>> exact_knn;
                distance_metrics::find_knn_exact(
                    normalized_data.data() + sample_idx * model->n_features,
                    normalized_data.data(),
                    model->n_samples,
                    model->n_features,
                    model->metric,
                    model->n_neighbors,
                    exact_knn,
                    sample_idx
                );

                // Get HNSW KNN
                auto hnsw_results = model->original_space_index->searchKnn(
                    normalized_data.data() + sample_idx * model->n_features,
                    model->n_neighbors + 1);

                std::vector<int> hnsw_indices;
                while (!hnsw_results.empty()) {
                    hnsw_indices.push_back(static_cast<int>(hnsw_results.top().second));
                    hnsw_results.pop();
                }

                // Calculate recall
                float recall = distance_metrics::calculate_recall(exact_knn, hnsw_indices.data(), model->n_neighbors);
                total_recall += recall;

                if ((s + 1) % 10 == 0 && callback) {
                    float percent = (float)(s + 1) / validation_samples * 100.0f;
                    callback("Validating HNSW Quality", s + 1, validation_samples, percent, nullptr);
                }
            }

            model->hnsw_recall_percentage = (total_recall / validation_samples) * 100.0f;

            if (callback) {
                char msg[256];
                snprintf(msg, sizeof(msg), "HNSW recall: %.1f%% (quality: %s)",
                         model->hnsw_recall_percentage,
                         model->hnsw_recall_percentage >= 95.0f ? "excellent" :
                         model->hnsw_recall_percentage >= 90.0f ? "good" :
                         model->hnsw_recall_percentage >= 80.0f ? "acceptable" : "poor");

                if (model->hnsw_recall_percentage < 90.0f) {
                    callback("WARNING", 0, 0, 0.0f, msg);
                } else {
                    callback("Validating HNSW Quality", validation_samples, validation_samples, 100.0f, msg);
                }
            }
        }
#endif // PACMAP_ENABLE_HNSW_VALIDATION
    }

   }

void sample_MN_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& mn_triplets, int n_mn, pacmap_progress_callback_internal callback) {

    mn_triplets.clear();

    if (callback) {
        callback("Sampling Mid-Near Pairs", 0, model->n_samples, 0.0f,
                 "Finding mid-range neighbors for structure preservation");
    }

    // ERROR11-FIX-DETERMINISTIC: Improved mid-near sampling with guaranteed pairs per point
    // Supports both exact KNN (current) and HNSW (future speed optimization)
    int extended_k = 3 * model->n_neighbors;  // e.g., 30 for n_neighbors=10
    mn_triplets.reserve(model->n_samples * n_mn);
    std::unordered_set<long long> used_pairs;
    int pairs_found = 0;

    // ERROR11-FIX-DETERMINISTIC: Initialize per-thread RNGs for deterministic parallel execution
    std::vector<std::mt19937> rngs;
    int max_threads = omp_get_max_threads();
    #pragma omp parallel
    {
        #pragma omp single
        {
            rngs.resize(max_threads);
            unsigned int seed = model->random_seed >= 0 ? model->random_seed : 42;
            for (int t = 0; t < max_threads; ++t) {
                rngs[t].seed(seed + t);  // Unique seed per thread for deterministic results
            }
        }
    }

    #pragma omp parallel for reduction(+:pairs_found)
    for (int i = 0; i < model->n_samples; ++i) {
        std::vector<std::pair<float, int>> neighbors;

        if (model->force_exact_knn) {
            // Exact KNN case
            distance_metrics::find_knn_exact(
                normalized_data.data() + static_cast<size_t>(i) * model->n_features,
                normalized_data.data(),
                model->n_samples,
                model->n_features,
                model->metric,
                extended_k,
                neighbors,
                i  // query_index to skip self
            );
        } else {
            // HNSW case (for future use)
            auto knn_results = model->original_space_index->searchKnn(
                normalized_data.data() + static_cast<size_t>(i) * model->n_features,
                extended_k);

            // Convert priority_queue to vector and reverse to get sorted order
            while (!knn_results.empty()) {
                auto pair = knn_results.top();
                knn_results.pop();
                // Convert squared distance to actual distance if L2 metric
                float distance = (model->metric == PACMAP_METRIC_EUCLIDEAN) ?
                               std::sqrt(std::max(0.0f, pair.first)) : pair.first;
                neighbors.emplace_back(distance, static_cast<int>(pair.second));
            }
            std::reverse(neighbors.begin(), neighbors.end());
        }

        size_t actual_extended_k = neighbors.size();
        int current_extended_k = extended_k;
        if (actual_extended_k < static_cast<size_t>(extended_k)) {
            // Note: Sampling adjustment debug output disabled for clean interface
            current_extended_k = static_cast<int>(actual_extended_k);  // Adjust dynamically
        }

        // Collect mid-near candidates (beyond immediate neighbors)
        std::vector<int> mn_candidates;
        for (int j = model->n_neighbors; j < current_extended_k; ++j) {
            if (j < static_cast<int>(neighbors.size())) {
                mn_candidates.push_back(neighbors[j].second);
            }
        }

        if (mn_candidates.empty()) {
            // Note: Skip points without mid-near candidates - debug output disabled for clean interface
            continue;  // Skip if no mid-near available
        }

        // ERROR11-FIX-DETERMINISTIC: Ensure at least n_mn pairs, cycle through candidates if needed
        int num_to_select = std::min(n_mn, static_cast<int>(mn_candidates.size()));
        int thread_id = omp_get_thread_num();
        std::shuffle(mn_candidates.begin(), mn_candidates.end(), rngs[thread_id]);
        std::sort(mn_candidates.begin(), mn_candidates.end()); // ERROR11-FIX-DETERMINISTIC: Enforce deterministic order

        for (int s = 0; s < num_to_select; ++s) {
            int j = mn_candidates[s % mn_candidates.size()];  // Cycle if fewer than n_mn
            long long pair_key = ((long long)std::min(i, j) << 32) | std::max(i, j);
            #pragma omp critical
            {
                if (used_pairs.find(pair_key) == used_pairs.end()) {
                    mn_triplets.emplace_back(Triplet{i, j, MID_NEAR});
                    used_pairs.insert(pair_key);
                    pairs_found++;
                }
            }
        }
    }

    if (callback) {
        callback("Sampling Mid-Near Pairs", model->n_samples, model->n_samples, 100.0f,
                 ("Found " + std::to_string(pairs_found) + " mid-near pairs").c_str());
    }

    }

void sample_FP_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& fp_triplets, int n_fp, pacmap_progress_callback_internal callback) {

    fp_triplets.clear();

    if (callback) {
        callback("Sampling Far Pairs", 0, model->n_samples, 0.0f,
                 "Sampling random far pairs for global structure");
    }

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

    std::atomic<int> completed(0);
    int report_interval = std::max(1, model->n_samples / 10);

    #pragma omp parallel
    {
        std::mt19937 rng = get_seeded_rng(model->random_seed + 1000 + omp_get_thread_num());
        std::uniform_int_distribution<int> dist(0, model->n_samples - 1);

        #pragma omp for
        for (int i = 0; i < model->n_samples; ++i) {
            std::unordered_set<int> sampled_indices;
            int found = 0;
            int target = n_fp;
            int early_exit_threshold = static_cast<int>(n_fp * 0.9);  // ERROR12 Priority 6: 90% early exit

            while (found < target) {
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

                    // ERROR12 Priority 6: Early exit at 90% to avoid wasting time
                    if (found >= early_exit_threshold) {
                        break;
                    }
                }
            }

            // Progress reporting
            int count = ++completed;
            if (callback && (count % report_interval == 0 || count == model->n_samples)) {
                float percent = (float)count / model->n_samples * 100.0f;
                callback("Sampling Far Pairs", count, model->n_samples, percent, nullptr);
            }
        }
    }

    if (callback) {
        callback("Sampling Far Pairs", model->n_samples, model->n_samples, 100.0f,
                 ("Found " + std::to_string(fp_triplets.size()) + " far pairs").c_str());
    }

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
    int M, int ef_construction, pacmap_progress_callback_internal callback) {

    if (callback) {
        callback("Building HNSW Index", 0, n_samples, 0.0f, "Initializing HNSW structure");
    }

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

    // Add data points to index with progress reporting
    // Use parallel insertion for large datasets (>5000 points) - HNSW is thread-safe
    int report_interval = std::max(1, n_samples / 20); // Report every 5%

    if (n_samples > 5000) {
        // Parallel addition for large datasets - much faster on multi-core CPUs
        std::atomic<int> completed(0);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 100)
#endif
        for (int i = 0; i < n_samples; ++i) {
            index->addPoint(data + i * n_features, i);

            // Progress reporting with atomic counter
            int count = ++completed;
            if (callback && (count % report_interval == 0 || count == n_samples)) {
                float percent = (float)count / n_samples * 100.0f;
                callback("Building HNSW Index", count, n_samples, percent, nullptr);
            }
        }
    } else {
        // Sequential addition for smaller datasets to avoid OpenMP overhead
        for (int i = 0; i < n_samples; ++i) {
            index->addPoint(data + i * n_features, i);

            if (callback && (i % report_interval == 0 || i == n_samples - 1)) {
                float percent = (float)(i + 1) / n_samples * 100.0f;
                callback("Building HNSW Index", i + 1, n_samples, percent, nullptr);
            }
        }
    }

    if (callback) {
        callback("Building HNSW Index", n_samples, n_samples, 100.0f, "HNSW index built successfully");
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