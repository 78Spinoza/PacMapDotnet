#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include "pacmap_utils.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <unordered_set>
#include <memory>
#include <iostream>
#include <atomic>

// ERROR12 Priority 5: Maximum vector size to prevent OOM with high ratios
#define MAX_VECTOR_SIZE (1ULL << 30)  // 1 billion elements max


void sample_triplets(PacMapModel* model, double* data, pacmap_progress_callback_internal callback) {

    
    // CRITICAL DEBUG: Add more detailed triplet analysis

    // Validate parameters first
    if (validate_parameters(model) != PACMAP_SUCCESS) {
        if (callback) callback("Error", 0, 100, 0.0f, "Parameter validation failed");
        return;
    }

    
    // Normalize/quantize data (reuse UMAP infrastructure) - NOW DOUBLE PRECISION
    std::vector<double> normalized_data;
    normalized_data.assign(data, data + model->n_samples * model->n_features);

    if (model->use_quantization) {
        // Apply quantization (reuse from existing infrastructure)
        // quantize_data(normalized_data, model->n_features);
    }

    // Save normalized data for transform - NOW DOUBLE PRECISION
    model->training_data = normalized_data;

    // Build HNSW index only if not using exact K-NN
    if (!model->force_exact_knn) {
        model->original_space_index = create_hnsw_index(normalized_data.data(),
                                                        model->n_samples,
                                                        model->n_features,
                                                        model->metric,
                                                        model->hnsw_m,
                                                        model->hnsw_ef_construction,
                                                        model->hnsw_ef_search,
                                                        model->random_seed,
                                                        callback);

        if (!model->original_space_index) {
            if (callback) callback("Error", 0, 100, 0.0f, "Failed to create HNSW index");
            return;
        }
    } else {
        if (callback) {
            callback("Exact KNN Mode", 100, 100, 100.0f, "Skipping HNSW index construction - using exact K-NN");
        }
    }

    // Initialize PCG RNG for deterministic behavior
    model->rng = get_seeded_pcg64(model->random_seed);

    // Sample three types of triplets
    std::vector<Triplet> neighbor_triplets, mn_triplets, fp_triplets;

    // Neighbor pairs using HNSW
    sample_neighbors_pair(model, normalized_data, neighbor_triplets, callback);

    // FIX19: Integer overflow fixes for large datasets (1M+ points)
    // Python reference (lines 169-174): n_MN = int(self.n_neighbors * self.MN_ratio)  // PER POINT, not total
    // Use int64_t to prevent overflow for large datasets
    int64_t n_mn_per_point = static_cast<int64_t>(model->n_neighbors * model->mn_ratio);
    int64_t n_mn = n_mn_per_point * model->n_samples;  // Total = per_point * n_samples
    // Cap to prevent infeasible targets (max possible pairs) - use int64_t throughout
    int64_t max_possible_pairs = static_cast<int64_t>(model->n_samples) * (model->n_samples - 1) / 2;
    n_mn = std::min(n_mn, std::min(max_possible_pairs, static_cast<int64_t>(MAX_VECTOR_SIZE)));

    
    sample_MN_pair(model, normalized_data, mn_triplets, n_mn, callback);

    // FIX19: Integer overflow fixes for large datasets (1M+ points)
    // Python reference (lines 173-174): n_FP = int(self.n_neighbors * self.FP_ratio)  // PER POINT, not total
    int64_t n_fp_per_point = static_cast<int64_t>(model->n_neighbors * model->fp_ratio);
    int64_t n_fp = n_fp_per_point * model->n_samples;  // Total = per_point * n_samples
    // Cap to prevent infeasible targets - use int64_t throughout
    n_fp = std::min(n_fp, std::min(max_possible_pairs, static_cast<int64_t>(MAX_VECTOR_SIZE)));

        sample_FP_pair(model, normalized_data, neighbor_triplets, fp_triplets, n_fp, callback);

    // MEMORY FIX: Combine all triplets using flat storage to prevent allocation failures
    model->clear_triplets();
    // Remove aggressive reservation for large datasets to avoid contiguous allocation failures
    size_t total_triplets = neighbor_triplets.size() + mn_triplets.size() + fp_triplets.size();
    if (model->n_samples <= 20000) {
        // Only reserve for smaller datasets where allocation won't fail
        model->triplets_flat.reserve(std::min(total_triplets * 3, static_cast<size_t>(MAX_VECTOR_SIZE)));
    }
    // Add triplets using flat storage
    for (const auto& t : neighbor_triplets) {
        model->add_triplet(static_cast<uint32_t>(t.anchor),
                          static_cast<uint32_t>(t.neighbor),
                          static_cast<uint32_t>(t.type));
    }
    for (const auto& t : mn_triplets) {
        model->add_triplet(static_cast<uint32_t>(t.anchor),
                          static_cast<uint32_t>(t.neighbor),
                          static_cast<uint32_t>(t.type));
    }
    for (const auto& t : fp_triplets) {
        model->add_triplet(static_cast<uint32_t>(t.anchor),
                          static_cast<uint32_t>(t.neighbor),
                          static_cast<uint32_t>(t.type));
    }

    // [OK] v2.8.9 FIX: DISABLED shuffling to match Python sequential processing
    // PROBLEM: Shuffled order + OpenMP = mixed force application = fragmentation
    // SOLUTION: Use Python's deterministic NEIGHBORMNFURTHER sequential order
        // std::shuffle(model->triplets.begin(), model->triplets.end(), model->rng);  // DISABLED - v2.8.9

    // ERROR13 FIX: Add triplet validation for flat storage
    // Remove invalid triplets (self-pairs, out-of-bounds indices)
    size_t before_filter = model->get_triplet_count();
    filter_invalid_triplets_flat(model->triplets_flat, model->n_samples);

    // FIX v2.5.9: CRITICAL - Python reference does NOT deduplicate directional pairs!
    // Python creates n_samples  n_neighbors directional neighbor triplets and uses ALL of them
    // Deduplicating (i,j) and (j,i) as the same pair was removing ~44K neighbor triplets
    // This caused severe underrepresentation of local attractive forces  oval formation
    //
    // DISABLE deduplication to match Python's behavior:
    
    // Keep this for reference - shows what deduplication was doing:
    // deduplicate_triplets(model->triplets);  //  COMMENTED OUT - was removing 44K neighbor triplets!

    size_t after_filter = model->get_triplet_count();
    if (callback && before_filter != after_filter) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Filtered %zu invalid triplets",
                 before_filter - after_filter);
        callback("Triplet Validation", 100, 100, 100.0f, msg);
    }


    // CRITICAL DEBUG: Analyze triplet quality
    if (model->get_triplet_count() > 0) {
        // Check if triplets have good diversity in indices
        std::unordered_set<int> unique_anchors, unique_neighbors;
        std::unordered_map<TripletType, int> type_counts;

        // MEMORY FIX: Iterate through flat triplet storage
        for (size_t i = 0; i < model->triplets_flat.size(); i += 3) {
            uint32_t anchor = model->triplets_flat[i];
            uint32_t neighbor = model->triplets_flat[i + 1];
            uint32_t type = model->triplets_flat[i + 2];

            unique_anchors.insert(static_cast<int>(anchor));
            unique_neighbors.insert(static_cast<int>(neighbor));
            type_counts[static_cast<TripletType>(type)]++;
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

            // MEMORY FIX: Iterate through flat triplet storage
            for (size_t i = 0; i < model->triplets_flat.size(); i += 3) {
                uint32_t triplet_type = model->triplets_flat[i + 2];
                if (triplet_type != static_cast<uint32_t>(type)) continue;

                uint32_t anchor = model->triplets_flat[i];
                uint32_t neighbor = model->triplets_flat[i + 1];

                double dist_double = distance_metrics::compute_distance(
                    normalized_data.data() + anchor * model->n_features,
                    normalized_data.data() + neighbor * model->n_features,
                    model->n_features, model->metric);
                float dist = static_cast<float>(dist_double);

                min_dist = std::min(min_dist, dist);
                max_dist = std::max(max_dist, dist);
                total_dist += dist;
                valid_samples++;
            }

            // Distance statistics output disabled for clean interface
        }
    }

    // Update triplet counts
    model->total_triplets = static_cast<int>(model->get_triplet_count());
    model->neighbor_triplets = static_cast<int>(neighbor_triplets.size());
    model->mid_near_triplets = static_cast<int>(mn_triplets.size());
    model->far_triplets = static_cast<int>(fp_triplets.size());

        // Triplet sampling completed - detailed output disabled for clean interface
    if (callback) {
        callback("Sampling Triplets", 100, 100, 100.0f, nullptr);
    }
}

void sample_neighbors_pair(PacMapModel* model, const std::vector<double>& normalized_data,
                         std::vector<Triplet>& neighbor_triplets, pacmap_progress_callback_internal callback) {

    neighbor_triplets.clear();
    // MEMORY FIX: Remove aggressive reservation for large datasets to prevent allocation failures
    if (model->n_samples <= 20000) {
        neighbor_triplets.reserve(model->n_samples * model->n_neighbors);
    }
    // For large datasets, let the vector grow naturally

    if (callback) {
        callback("Sampling Neighbor Pairs", 0, model->n_samples, 0.0f,
                 "Using Python-matching scaled distance neighbor selection");
    }

    //  v2.8.11 CRITICAL FIX: PYTHON-MATCHING SCALED DISTANCE NEIGHBOR SELECTION
    // Python (pacmap.py lines 373-427): Fetches n_neighbors + 50 extra neighbors,
    // calculates sigma (local scale), scales distances by local density,
    // sorts by scaled distance, then picks top n_neighbors
    //
    // This density-aware selection is CRITICAL for local structure preservation!
    // Without it: dense regions over-connected, sparse regions under-connected
    // Result: Deformed mammoth with cut legs and poor local structure

    
    // STEP 1: Calculate n_neighbors_extra (Python: n_neighbors + 50)
    int n_neighbors_extra = static_cast<int>(std::min(model->n_neighbors + 50, model->n_samples - 1));

    // STEP 2: Fetch extra neighbors and calculate sigma for all points
    // Store KNN indices and distances for later scaling
    // MEMORY FIX: Use conservative allocation for large datasets to prevent STL vector allocation failure
    std::vector<std::vector<int>> knn_indices;
    std::vector<std::vector<double>> knn_distances;
    std::vector<double> sigma(model->n_samples);

    // For large datasets, build vectors incrementally to avoid massive allocation
    if (model->n_samples > 20000) {
        knn_indices.reserve(model->n_samples);
        knn_distances.reserve(model->n_samples);
        // Don't pre-reserve inner vectors - let them grow naturally
        for (int i = 0; i < model->n_samples; ++i) {
            knn_indices.emplace_back();
            knn_distances.emplace_back();
        }
    } else {
        // Use normal allocation for smaller datasets
        knn_indices.resize(model->n_samples);
        knn_distances.resize(model->n_samples);
    }

    int report_interval = std::max(50, static_cast<int>(model->n_samples / 100)); // Report every 1% (min 50 samples for performance)
    std::atomic<int> completed(0);

    if (callback) {
        callback("Fetching Extra Neighbors", 0, model->n_samples, 0.0f,
                 model->force_exact_knn ? "Using exact KNN" : "Using HNSW");
    }

    // Fetch neighbors and calculate sigma
    #pragma omp parallel for if(model->n_samples > 1000)
    for (int i = 0; i < model->n_samples; ++i) {
        std::vector<std::pair<double, int>> knn_temp;

        if (model->force_exact_knn) {
            // Exact K-NN mode
            std::vector<std::pair<float, int>> knn_float;
            distance_metrics::find_knn_exact(
                normalized_data.data() + i * model->n_features,
                normalized_data.data(),
                model->n_samples,
                model->n_features,
                model->metric,
                n_neighbors_extra + 1,  // +1 to include self
                knn_float,
                i  // query_index to skip self
            );

            // Convert to double and skip self
            for (int j = 1; j < static_cast<int>(knn_float.size()); ++j) {
                knn_temp.emplace_back(static_cast<double>(knn_float[j].first), knn_float[j].second);
            }
        } else {
            // HNSW mode
            std::vector<float> query_point_float(model->n_features);
            for (int d = 0; d < model->n_features; ++d) {
                query_point_float[d] = static_cast<float>(normalized_data[i * model->n_features + d]);
            }

            auto knn_results = model->original_space_index->searchKnn(
                query_point_float.data(),
                n_neighbors_extra + 1);  // +1 to include self

            // Convert priority_queue to vector and skip self
            std::vector<std::pair<float, size_t>> knn_hnsw;
            while (!knn_results.empty()) {
                knn_hnsw.push_back(knn_results.top());
                knn_results.pop();
            }
            std::reverse(knn_hnsw.begin(), knn_hnsw.end());

            // Skip self (first element) and convert to double
            for (int j = 1; j < static_cast<int>(knn_hnsw.size()); ++j) {
                knn_temp.emplace_back(static_cast<double>(knn_hnsw[j].first),
                                     static_cast<int>(knn_hnsw[j].second));
            }
        }

        // Store indices and distances
        // For both normal and incremental allocation, use the same approach
        knn_indices[i].clear();
        knn_distances[i].clear();
        for (size_t j = 0; j < knn_temp.size(); ++j) {
            knn_distances[i].push_back(knn_temp[j].first);
            knn_indices[i].push_back(knn_temp[j].second);
        }

        //  STEP 3: Calculate sigma (local scale) - Python line 423
        // Python: sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
        // Mean of distances to 4th, 5th, 6th nearest neighbors (indices 3, 4, 5)
        if (knn_distances[i].size() >= 6) {
            double mean_dist = (knn_distances[i][3] + knn_distances[i][4] + knn_distances[i][5]) / 3.0;
            sigma[i] = std::max(mean_dist, 1e-10);
        } else if (knn_distances[i].size() >= 3) {
            // Fallback for small datasets: use available neighbors
            double sum = 0.0;
            int start_idx = std::min(3, static_cast<int>(knn_distances[i].size()) - 1);
            for (int j = start_idx; j < static_cast<int>(knn_distances[i].size()); ++j) {
                sum += knn_distances[i][j];
            }
            sigma[i] = std::max(sum / (knn_distances[i].size() - start_idx), 1e-10);
        } else {
            // Emergency fallback
            sigma[i] = 1.0;
        }

        // Progress reporting
        int count = ++completed;
        if (callback && (count % report_interval == 0 || count == model->n_samples)) {
            float percent = (float)count / model->n_samples * 100.0f;
            callback("Fetching Extra Neighbors", count, model->n_samples, percent, nullptr);
        }
    }

    if (callback) {
        callback("Scaling Distances", 0, model->n_samples, 0.0f,
                 "Applying density-aware distance scaling");
    }

    //  STEP 4: Scale distances and sort by scaled distance
    completed = 0;
    // TEMP FIX: Force single thread to test cross pattern issue
    #pragma omp parallel for if(model->n_samples > 1000)
    for (int i = 0; i < model->n_samples; ++i) {
        // Scale distances by local density - Python lines 236-239 (now 142-151)
        // scaled_dist[i, j] = d / (sig[i] * sig[j])
        std::vector<std::pair<double, int>> scaled_dist_with_idx;
        scaled_dist_with_idx.reserve(knn_indices[i].size());

        for (size_t j = 0; j < knn_indices[i].size(); ++j) {
            int neighbor_idx = knn_indices[i][j];
            double d_ij = knn_distances[i][j];
            double d_ij_sq = d_ij * d_ij;
            double scaled_d = d_ij_sq / (sigma[i] * sigma[neighbor_idx]);
            scaled_dist_with_idx.emplace_back(scaled_d, neighbor_idx);
        }

        //  STEP 5: Sort by scaled distance - Python line 124
        // Python: scaled_sort = np.argsort(scaled_dist[i])
        std::sort(scaled_dist_with_idx.begin(), scaled_dist_with_idx.end());

        //  STEP 6: Pick top n_neighbors from scaled & sorted list
        #pragma omp critical
        {
            for (int j = 0; j < model->n_neighbors && j < static_cast<int>(scaled_dist_with_idx.size()); ++j) {
                int neighbor_idx = scaled_dist_with_idx[j].second;
                neighbor_triplets.emplace_back(i, neighbor_idx, NEIGHBOR);
            }
        }

        // Progress reporting
        int count = ++completed;
        if (callback && (count % report_interval == 0 || count == model->n_samples)) {
            float percent = (float)count / model->n_samples * 100.0f;
            callback("Scaling Distances", count, model->n_samples, percent, nullptr);
        }
    }

    if (callback) {
        callback("Sampling Neighbor Pairs", model->n_samples, model->n_samples, 100.0f,
                 "Neighbor pairs sampled using Python-matching scaled distance selection");
    }

       }

void sample_MN_pair(PacMapModel* model, const std::vector<double>& normalized_data,
                   std::vector<Triplet>& mn_triplets, int64_t n_mn, pacmap_progress_callback_internal callback) {

    mn_triplets.clear();

    if (callback) {
        callback("Sampling Mid-Near Pairs", 0, model->n_samples, 0.0f,
                 "CRITICAL FIX v2.8.14: Python-exact fixed-size array algorithm");
    }

    // � CRITICAL FIX v2.8.14: IMPLEMENT PYTHON'S EXACT ALGORITHM
    // Python reference: sample_MN_pair in pacmap.py lines 146-168
    // Previous C++ implementation was COMPLETELY WRONG!
    
    // Calculate per-point target exactly like Python - use int64_t for safety
    int64_t n_MN_per_point = n_mn / model->n_samples;

    // MEMORY FIX: Remove aggressive reservations for large datasets
    // Only reserve for smaller datasets where allocation won't fail
    if (model->n_samples <= 20000) {
        mn_triplets.reserve(model->n_samples * n_MN_per_point);
    }
    // For large datasets, let the vector grow naturally

    // Thread-safe PCG RNG for deterministic behavior
    pcg64_fast rng = get_seeded_pcg64(model->random_seed + 2000);
    std::uniform_int_distribution<int> dist(0, model->n_samples - 1);

    int report_interval = std::max(50, static_cast<int>(model->n_samples / 100)); // Report every 1% (min 50 samples for performance)

    //  PYTHON EXACT: Sequential processing like numba.prange (but without OpenMP to match Python exactly)
    for (int i = 0; i < model->n_samples; ++i) {
        //  PYTHON EXACT: For each j in range(n_MN) - fixed count per point
        for (int j = 0; j < n_MN_per_point; ++j) {
            //  PYTHON EXACT: sample_FP implementation with iterative rejection
            std::vector<int> sampled;

            // Sample 6 candidates with Python's exact rejection logic
            while (sampled.size() < 6) {
                int candidate = dist(rng);

                //  PYTHON EXACT: Self rejection
                if (candidate == i) continue;

                //  PYTHON EXACT: Duplicate rejection within current batch
                bool duplicate = false;
                for (int t = 0; t < static_cast<int>(sampled.size()); ++t) {
                    if (candidate == sampled[t]) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) continue;

                //  PYTHON EXACT: Iterative rejection using PREVIOUS pairs in this iteration
                // Python: reject_ind=pair_MN[i * n_MN:i * n_MN + j, 1]
                bool reject = false;
                for (int prev_j = 0; prev_j < j; ++prev_j) {
                    int prev_pair_idx = i * n_MN_per_point + prev_j;
                    // FIX: Check bounds correctly - we're building sequentially, so prev_pair_idx is always valid
                    if (mn_triplets.size() > prev_pair_idx &&
                        mn_triplets[prev_pair_idx].neighbor == candidate) {
                        reject = true;
                        break;
                    }
                }
                if (reject) continue;

                sampled.push_back(candidate);
            }

            //  PYTHON EXACT: Calculate distances to all 6 candidates
            std::vector<std::pair<double, int>> candidate_distances;
            candidate_distances.reserve(6);

            const double* point_i = normalized_data.data() + static_cast<size_t>(i) * model->n_features;

            for (int t = 0; t < 6; ++t) {
                const double* point_j = normalized_data.data() + static_cast<size_t>(sampled[t]) * model->n_features;
                double distance = distance_metrics::compute_distance(
                    point_i, point_j, model->n_features, model->metric);
                candidate_distances.emplace_back(distance, sampled[t]);
            }

            //  PYTHON EXACT: Find closest candidate and DISCARD it
            std::sort(candidate_distances.begin(), candidate_distances.end());

            //  PYTHON EXACT: Pick 2nd closest from remaining candidates
            int picked = candidate_distances[1].second;

            //  PYTHON EXACT: Add to results with NO deduplication
            mn_triplets.emplace_back(Triplet{i, picked, MID_NEAR});
        }

        // Progress reporting
        if (callback && ((i + 1) % report_interval == 0 || (i + 1) == model->n_samples)) {
            float percent = (float)(i + 1) / model->n_samples * 100.0f;
            char progress_msg[128];
            snprintf(progress_msg, sizeof(progress_msg), "Processed %d/%jd samples, generated %zu triplets",
                     i + 1, (intmax_t)model->n_samples, mn_triplets.size());
            callback("Sampling Mid-Near Pairs", i + 1, model->n_samples, percent, progress_msg);
        }
    }

    if (callback) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Generated %zu mid-near pairs using Python's EXACT algorithm (v2.8.14)",
                 mn_triplets.size());
        callback("Sampling Mid-Near Pairs", model->n_samples, model->n_samples, 100.0f, msg);
    }

        }

void sample_FP_pair(PacMapModel* model, const std::vector<double>& normalized_data,
                   const std::vector<Triplet>& neighbor_triplets,
                   std::vector<Triplet>& fp_triplets, int64_t n_fp, pacmap_progress_callback_internal callback) {

    fp_triplets.clear();

    if (callback) {
        callback("Sampling Far Pairs", 0, model->n_samples, 0.0f,
                 "CRITICAL FIX v2.8.15: Python-exact far pair sampling algorithm");
    }

    // CRITICAL FIX v2.8.15: IMPLEMENT PYTHON'S EXACT FAR PAIR SAMPLING
    // Python reference: sample_FP in pacmap.py lines 37-56
    // Previous C++ implementation had 4 CRITICAL discrepancies from Python!
    
    // Calculate per-point target exactly like Python - use int64_t for safety
    int64_t n_FP_per_point = n_fp / model->n_samples;

    // MEMORY FIX: Remove aggressive reservations for large datasets
    // Only reserve for smaller datasets where allocation won't fail
    if (model->n_samples <= 20000) {
        fp_triplets.reserve(model->n_samples * n_FP_per_point);
    }
    // For large datasets, let the vector grow naturally

    // FIX 1: Build UNIDIRECTIONAL neighbor exclusion sets (Python: only i->j, not j->i)
    // Python reference: reject_ind=pair_neighbors[i * n_neighbors:(i + 1) * n_neighbors, 1]
    // MEMORY FIX: For large datasets, use hash-based lookup instead of storing all sets
    std::vector<std::vector<int>> neighbor_lists;
    if (model->n_samples > 20000) {
        // For large datasets, use vector of vectors (more memory efficient than unordered_set)
        neighbor_lists.resize(model->n_samples);
        for (const auto& t : neighbor_triplets) {
            if (t.anchor < neighbor_lists.size()) {
                neighbor_lists[t.anchor].push_back(t.neighbor);
            }
        }
    } else {
        // For smaller datasets, use unordered_set for faster lookup
        std::vector<std::unordered_set<int>> neighbor_sets;
        neighbor_sets.resize(model->n_samples);
        for (const auto& t : neighbor_triplets) {
            neighbor_sets[t.anchor].insert(t.neighbor);
        }

        // Convert to vector format for uniform processing
        neighbor_lists.resize(model->n_samples);
        for (int i = 0; i < model->n_samples; ++i) {
            neighbor_lists[i].assign(neighbor_sets[i].begin(), neighbor_sets[i].end());
        }
    }

    int report_interval = std::max(50, static_cast<int>(model->n_samples / 100)); // Report every 1% (min 50 samples for performance)

    // PYTHON EXACT: Sequential processing like Python numba.prange
    for (int i = 0; i < model->n_samples; ++i) {
        // FIX 1: PER-POINT deterministic seeding (Python: per-point RNG seeding)
        // Python uses deterministic per-point seeding, not thread-local seeding
        pcg64_fast rng = get_seeded_pcg64(model->random_seed + 3000 + i);
        std::uniform_int_distribution<int> dist(0, model->n_samples - 1);

        std::unordered_set<int> sampled_indices;
        int found = 0;
        int attempts = 0;
        const int max_attempts = model->n_samples * 10;  // Safety limit to prevent infinite loops

        // PYTHON EXACT: Continue until target reached with safety check
        while (found < n_FP_per_point && attempts < max_attempts) {
            int j = dist(rng);
            attempts++;

            // PYTHON EXACT: Rejection logic matches Python exactly
            bool is_neighbor = false;
            for (int neighbor : neighbor_lists[i]) {
                if (neighbor == j) {
                    is_neighbor = true;
                    break;
                }
            }

            if (j != i &&
                sampled_indices.find(j) == sampled_indices.end() &&
                !is_neighbor) {

                // FIX 2: REMOVED global deduplication (Python allows duplicate pairs)
                // Python does NOT deduplicate far pairs - it allows the same pair to appear multiple times
                fp_triplets.emplace_back(Triplet{i, j, FURTHER});
                found++;
                sampled_indices.insert(j);
            }
        }

        // SAFETY WARNING: If we couldn't find enough unique candidates
        if (found < n_FP_per_point) {
                    }

        // Progress reporting
        if (callback && ((i + 1) % report_interval == 0 || (i + 1) == model->n_samples)) {
            float percent = (float)(i + 1) / model->n_samples * 100.0f;
            char progress_msg[128];
            snprintf(progress_msg, sizeof(progress_msg), "Processed %d/%jd samples, generated %zu triplets",
                     i + 1, (intmax_t)model->n_samples, fp_triplets.size());
            callback("Sampling Far Pairs", i + 1, model->n_samples, percent, progress_msg);
        }
    }

    if (callback) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Generated %zu far pairs using Python's EXACT algorithm (v2.8.15)",
                 fp_triplets.size());
        callback("Sampling Far Pairs", model->n_samples, model->n_samples, 100.0f, msg);
    }

        }

void distance_based_sampling(PacMapModel* model, const std::vector<double>& data,
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
        double dist_double = distance_metrics::compute_distance(data.data() + i * model->n_features,
                                                 data.data() + j * model->n_features,
                                                 model->n_features, model->metric);
        float distance = static_cast<float>(dist_double);

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
    const double* data, int n_samples, int n_features, PacMapMetric metric,
    int M, int ef_construction, int ef_search, int random_seed, pacmap_progress_callback_internal callback) {

    if (callback) {
        callback("Building HNSW Index", 0, n_samples, 0.0f, "Initializing HNSW structure");
    }

    // Convert double data to float for HNSW (HNSW only supports float*)
    std::vector<float> data_float(n_samples * n_features);
    for (size_t i = 0; i < n_samples * n_features; ++i) {
        data_float[i] = static_cast<float>(data[i]);
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

    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space, n_samples, M, ef_construction, random_seed);  // Use user-provided random seed
    index->setEf(ef_search);  // Set query-time ef parameter to match API specification

    // Add data points to index with progress reporting
    // Use parallel insertion for large datasets (>5000 points) - HNSW is thread-safe
    int report_interval = std::max(50, n_samples / 100); // Report every 1% (min 50 samples for performance)

    if (n_samples > 5000) {
        // Parallel addition for large datasets - much faster on multi-core CPUs
        std::atomic<int> completed(0);

        // TEMP FIX: Force single thread to test cross pattern issue
        #ifdef _OPENMP
           #pragma omp parallel for schedule(dynamic, 100)
        #endif
        for (int i = 0; i < n_samples; ++i) {
            index->addPoint(data_float.data() + i * n_features, i);

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
            index->addPoint(data_float.data() + i * n_features, i);

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

void filter_invalid_triplets_flat(std::vector<uint32_t>& triplets, int n_samples) {
    // MEMORY FIX: Filter invalid triplelets in flat storage format
    // Format: [anchor1, neighbor1, type1, anchor2, neighbor2, type2, ...]

    size_t write_pos = 0;
    for (size_t read_pos = 0; read_pos < triplets.size(); read_pos += 3) {
        if (read_pos + 2 >= triplets.size()) break; // Ensure we have complete triplet

        uint32_t anchor = triplets[read_pos];
        uint32_t neighbor = triplets[read_pos + 1];
        uint32_t type = triplets[read_pos + 2];

        // Validate triplet
        bool is_valid = (anchor < static_cast<uint32_t>(n_samples) &&
                        neighbor < static_cast<uint32_t>(n_samples) &&
                        anchor != neighbor &&
                        type <= 2); // Valid TripletType

        if (is_valid) {
            // Move valid triplet to write position
            if (write_pos != read_pos) {
                triplets[write_pos] = anchor;
                triplets[write_pos + 1] = neighbor;
                triplets[write_pos + 2] = type;
            }
            write_pos += 3;
        }
    }

    // Resize to remove invalid triplets
    triplets.resize(write_pos);
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

// Legacy wrapper - kept for compatibility
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

// � PHASE 3: ENHANCED TRIPLET QUALITY VALIDATION
void validate_triplet_quality(const std::vector<Triplet>& triplets,
                            const std::vector<double>& embedding, int n_components,
                            pacmap_progress_callback_internal callback) {
    if (triplets.empty()) {
        if (callback) callback("Triplet Validation", 0, 0, 0.0f, " No triplets to validate");
        return;
    }

    // � Validate triplet structure and quality metrics
    std::unordered_map<TripletType, int> type_counts;
    std::unordered_set<int> unique_anchors, unique_neighbors;
    int self_pairs = 0, out_of_bounds = 0;

    for (const auto& t : triplets) {
        type_counts[t.type]++;
        unique_anchors.insert(t.anchor);
        unique_neighbors.insert(t.neighbor);

        // Validate indices
        if (t.anchor == t.neighbor) self_pairs++;
        if (t.anchor < 0 || t.neighbor < 0) out_of_bounds++;
    }

    // � Coverage analysis
    float anchor_coverage = (float)unique_anchors.size() / embedding.size() * 100.0f / n_components;
    float neighbor_coverage = (float)unique_neighbors.size() / embedding.size() * 100.0f / n_components;

    char validation_msg[1024];
    snprintf(validation_msg, sizeof(validation_msg),
            "� TRIPLET VALIDATION v2.8.10: Total=%zu | Types: N=%d, MN=%d, F=%d | Coverage: anchors=%.1f%%, neighbors=%.1f%% | Issues: self=%d, oob=%d",
            triplets.size(), type_counts[NEIGHBOR], type_counts[MID_NEAR], type_counts[FURTHER],
            anchor_coverage, neighbor_coverage, self_pairs, out_of_bounds);

    if (callback) callback("Triplet Validation", 0, 0, 0.0f, validation_msg);

    //  Check for potential issues
    if (self_pairs > 0) {
        char warning_msg[512];
            }

    if (anchor_coverage < 50.0f || neighbor_coverage < 50.0f) {
        char coverage_warning[512];
            }
}

void analyze_triplet_distance_distributions(const std::vector<Triplet>& triplets,
                                          const std::vector<double>& embedding, int n_components,
                                          pacmap_progress_callback_internal callback) {
    if (triplets.empty()) return;

    // � Analyze distance distributions by triplet type
    std::unordered_map<TripletType, std::vector<double>> distances_by_type;

    for (const auto& t : triplets) {
        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // Compute Euclidean distance
        double d_ij_squared = 0.0;
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij_squared += diff * diff;
        }
        double d_ij = std::sqrt(std::max(d_ij_squared, 1e-15));

        distances_by_type[t.type].push_back(d_ij);
    }

    // Calculate statistics for each type
    for (auto const& [type, distances] : distances_by_type) {
        if (distances.empty()) continue;

        auto [min_it, max_it] = std::minmax_element(distances.begin(), distances.end());
        double min_dist = *min_it, max_dist = *max_it;
        double mean_dist = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

        // Calculate standard deviation
        double variance = 0.0;
        for (double d : distances) {
            variance += (d - mean_dist) * (d - mean_dist);
        }
        double std_dev = std::sqrt(variance / distances.size());

        const char* type_name = "";
        switch (type) {
            case NEIGHBOR: type_name = "NEIGHBOR"; break;
            case MID_NEAR: type_name = "MID_NEAR"; break;
            case FURTHER: type_name = "FURTHER"; break;
        }

        char distance_msg[1024];
            }

    // � Validate distance ordering: expect NEIGHBOR < MID_NEAR < FURTHER
    if (distances_by_type[NEIGHBOR].size() > 0 && distances_by_type[MID_NEAR].size() > 0 && distances_by_type[FURTHER].size() > 0) {
        double n_mean = std::accumulate(distances_by_type[NEIGHBOR].begin(), distances_by_type[NEIGHBOR].end(), 0.0) / distances_by_type[NEIGHBOR].size();
        double mn_mean = std::accumulate(distances_by_type[MID_NEAR].begin(), distances_by_type[MID_NEAR].end(), 0.0) / distances_by_type[MID_NEAR].size();
        double f_mean = std::accumulate(distances_by_type[FURTHER].begin(), distances_by_type[FURTHER].end(), 0.0) / distances_by_type[FURTHER].size();

        if (n_mean >= mn_mean || mn_mean >= f_mean) {
            char ordering_warning[512];
                    }
    }
}

void check_triplet_coverage(const std::vector<Triplet>& triplets, int n_samples,
                          pacmap_progress_callback_internal callback) {
    if (triplets.empty()) return;

    // � Analyze how well triplets cover the dataset
    std::vector<int> anchor_frequency(n_samples, 0);
    std::vector<int> neighbor_frequency(n_samples, 0);
    std::unordered_map<TripletType, std::unordered_set<int>> type_coverage;

    for (const auto& t : triplets) {
        anchor_frequency[t.anchor]++;
        neighbor_frequency[t.neighbor]++;
        type_coverage[t.type].insert(t.anchor);
        type_coverage[t.type].insert(t.neighbor);
    }

    // Calculate coverage statistics
    int anchors_with_triplets = 0, neighbors_with_triplets = 0;
    double avg_anchor_freq = 0.0, avg_neighbor_freq = 0.0;

    for (int i = 0; i < n_samples; ++i) {
        if (anchor_frequency[i] > 0) anchors_with_triplets++;
        if (neighbor_frequency[i] > 0) neighbors_with_triplets++;
        avg_anchor_freq += anchor_frequency[i];
        avg_neighbor_freq += neighbor_frequency[i];
    }
    avg_anchor_freq /= n_samples;
    avg_neighbor_freq /= n_samples;

    float anchor_coverage_pct = (float)anchors_with_triplets / n_samples * 100.0f;
    float neighbor_coverage_pct = (float)neighbors_with_triplets / n_samples * 100.0f;

    char coverage_msg[1024];
    
    // Check by type
    for (auto const& [type, coverage] : type_coverage) {
        const char* type_name = "";
        switch (type) {
            case NEIGHBOR: type_name = "NEIGHBOR"; break;
            case MID_NEAR: type_name = "MID_NEAR"; break;
            case FURTHER: type_name = "FURTHER"; break;
        }

        float type_coverage_pct = (float)coverage.size() / n_samples * 100.0f;
        char type_msg[512];
        snprintf(type_msg, sizeof(type_msg), "  %s type coverage: %.1f%% (%zu unique points)",
                type_name, type_coverage_pct, coverage.size());
        if (callback) callback("Coverage Analysis", 0, 0, 0.0f, type_msg);
    }

    // Warn about poor coverage
    if (anchor_coverage_pct < 80.0f) {
        char warning[512];
            }
}

void detect_triplet_anomalies(const std::vector<Triplet>& triplets,
                             const std::vector<double>& embedding, int n_components,
                             pacmap_progress_callback_internal callback) {
    if (triplets.empty()) return;

    // � Detect potential anomalies in triplet structure
    std::unordered_map<long long, int> pair_frequency;
    std::vector<int> point_triplet_count(embedding.size() / n_components, 0);
    int duplicate_pairs = 0, highly_connected_points = 0;
    const int max_expected_connections = 50;  // Threshold for "highly connected"

    // Count pair frequencies and point connections
    for (const auto& t : triplets) {
        long long pair_key = ((long long)std::min(t.anchor, t.neighbor) << 32) | std::max(t.anchor, t.neighbor);
        pair_frequency[pair_key]++;

        point_triplet_count[t.anchor]++;
        point_triplet_count[t.neighbor]++;

        if (pair_frequency[pair_key] > 1) {
            duplicate_pairs++;
        }
    }

    // Count highly connected points
    for (int count : point_triplet_count) {
        if (count > max_expected_connections) {
            highly_connected_points++;
        }
    }

    char anomaly_msg[1024];
    
    // Check for extreme cases
    if (duplicate_pairs > triplets.size() * 0.1) {
        char warning[512];
            }
}