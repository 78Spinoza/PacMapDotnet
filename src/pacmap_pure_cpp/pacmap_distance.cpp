#include "pacmap_distance.h"
#include "pacmap_utils.h"  // For safe multiplication functions
#include <cstdio>  // For fprintf warnings

namespace distance_metrics {

    // FIX17.md Step 6: Inline distance functions are in header, but compute_distance functions remain here
    // (switch statements are not ideal for inlining)

    float compute_distance(const float* a, const float* b, int dim, PacMapMetric metric) {
        switch (metric) {
        case PACMAP_METRIC_EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case PACMAP_METRIC_COSINE:
            return cosine_distance(a, b, dim);
        case PACMAP_METRIC_MANHATTAN:
            return manhattan_distance(a, b, dim);
        case PACMAP_METRIC_CORRELATION:
            return correlation_distance(a, b, dim);
        case PACMAP_METRIC_HAMMING:
            return hamming_distance(a, b, dim);
        default:
            return euclidean_distance(a, b, dim);
        }
    }

    double compute_distance(const double* a, const double* b, int dim, PacMapMetric metric) {
        switch (metric) {
        case PACMAP_METRIC_EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case PACMAP_METRIC_COSINE:
            return cosine_distance(a, b, dim);
        case PACMAP_METRIC_MANHATTAN:
            return manhattan_distance(a, b, dim);
        case PACMAP_METRIC_CORRELATION:
            return correlation_distance(a, b, dim);
        case PACMAP_METRIC_HAMMING:
            return hamming_distance(a, b, dim);
        default:
            return euclidean_distance(a, b, dim);
        }
    }

    // Data validation functions for specific metrics
    bool validate_hamming_data(const float* data, int n_obs, int n_dim) {
        int non_binary_count = 0;
        const int MAX_NON_BINARY_TO_CHECK = std::min(1000, n_obs); // Sample validation
        const int MAX_FEATURES_TO_CHECK = std::min(50, n_dim);     // Sample features

        for (int i = 0; i < MAX_NON_BINARY_TO_CHECK; i++) {
            for (int j = 0; j < MAX_FEATURES_TO_CHECK; j++) {
                float val = data[i * n_dim + j];
                // Check if value is approximately 0 or 1 (allowing small floating point errors)
                if (!(std::abs(val) < 1e-6f || std::abs(val - 1.0f) < 1e-6f)) {
                    non_binary_count++;
                    if (non_binary_count > 10) { // Stop early if clearly not binary
                        return false;
                    }
                }
            }
        }

        // Consider data binary if less than 5% non-binary values in sample
        float non_binary_ratio = static_cast<float>(non_binary_count) / (MAX_NON_BINARY_TO_CHECK * MAX_FEATURES_TO_CHECK);
        return non_binary_ratio < 0.05f;
    }

    bool validate_correlation_data(const float* data, int n_obs, int n_dim) {
        // Check if data has sufficient variance for meaningful correlation
        if (n_dim < 2) return false; // Correlation needs at least 2 dimensions

        // Sample a few features to check for constant values (zero variance)
        const int MAX_FEATURES_TO_CHECK = std::min(10, n_dim);
        int constant_features = 0;

        for (int feature = 0; feature < MAX_FEATURES_TO_CHECK; feature++) {
            float min_val = data[feature];
            float max_val = data[feature];

            // Sample values to check variance
            int sample_size = std::min(100, n_obs);
            for (int i = 0; i < sample_size; i++) {
                float val = data[i * n_dim + feature];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            // Check if feature has essentially no variance
            if ((max_val - min_val) < 1e-10f) {
                constant_features++;
            }
        }

        // Warn if more than 50% of sampled features are constant
        return constant_features < (MAX_FEATURES_TO_CHECK / 2);
    }

    // Main validation function that issues warnings for inappropriate data
    void validate_metric_data(const float* data, int n_obs, int n_dim, PacMapMetric metric) {
        switch (metric) {
            case PACMAP_METRIC_HAMMING:
                if (!validate_hamming_data(data, n_obs, n_dim)) {
                    fprintf(stderr, "WARNING: Hamming metric expects binary data (0/1 values). "
                                   "Non-binary data detected - results may be meaningless.\n");
                }
                break;

            case PACMAP_METRIC_CORRELATION:
                if (!validate_correlation_data(data, n_obs, n_dim)) {
                    fprintf(stderr, "WARNING: Correlation metric expects data with meaningful variance. "
                                   "Constant or near-constant features detected - results may be unreliable.\n");
                }
                break;

            case PACMAP_METRIC_COSINE:
                // Could add validation for zero-norm vectors, but cosine_distance already handles this
                break;

            case PACMAP_METRIC_EUCLIDEAN:
            case PACMAP_METRIC_MANHATTAN:
            default:
                // These metrics are generally robust to different data types
                break;
        }
    }

    // Find k nearest neighbors from a query point to all dataset points (exact search)
    void find_knn_exact(const float* query_point, const float* dataset, int n_obs, int n_dim,
                       PacMapMetric metric, int k_neighbors, std::vector<std::pair<float, int>>& neighbors_out,
                       int query_index) {

        neighbors_out.clear();
        neighbors_out.reserve(k_neighbors);

        // If query_index is provided, we need to skip the self-match
        bool skip_self = (query_index >= 0);

        // Use partial_sort to efficiently find k smallest distances
        std::vector<std::pair<float, int>> all_distances;
        all_distances.reserve(n_obs);

        // Compute distances to all points
        for (int i = 0; i < n_obs; ++i) {
            // Skip self-match if query_index is provided
            if (skip_self && i == query_index) {
                continue;
            }

            float dist = compute_distance(query_point, dataset + i * n_dim, n_dim, metric);
            all_distances.emplace_back(dist, i);
        }

        // Find k smallest distances using partial_sort for efficiency
        int k_actual = std::min(k_neighbors, (int)all_distances.size());
        if (k_actual > 0) {
            std::partial_sort(all_distances.begin(), all_distances.begin() + k_actual, all_distances.end());

            // Copy k smallest to output
            neighbors_out.assign(all_distances.begin(), all_distances.begin() + k_actual);
        }
    }

    // Double precision version of find_knn_exact
    void find_knn_exact(const double* query_point, const double* dataset, int n_obs, int n_dim,
                       PacMapMetric metric, int k_neighbors, std::vector<std::pair<float, int>>& neighbors_out,
                       int query_index) {

        neighbors_out.clear();
        neighbors_out.reserve(k_neighbors);

        // If query_index is provided, we need to skip the self-match
        bool skip_self = (query_index >= 0);

        // Use partial_sort to efficiently find k smallest distances
        std::vector<std::pair<float, int>> all_distances;
        all_distances.reserve(n_obs);

        // Compute distances to all points
        for (int i = 0; i < n_obs; ++i) {
            // Skip self-match if query_index is provided
            if (skip_self && i == query_index) {
                continue;
            }

            double dist = compute_distance(query_point, dataset + i * n_dim, n_dim, metric);
            all_distances.emplace_back(static_cast<float>(dist), i);  // Store as float for consistency
        }

        // Find k smallest distances using partial_sort for efficiency
        int k_actual = std::min(k_neighbors, (int)all_distances.size());
        if (k_actual > 0) {
            std::partial_sort(all_distances.begin(), all_distances.begin() + k_actual, all_distances.end());

            // Copy k smallest to output
            neighbors_out.assign(all_distances.begin(), all_distances.begin() + k_actual);
        }
    }

    // Compare two neighbor lists and calculate recall (intersection / union)
    float calculate_recall(const std::vector<std::pair<float, int>>& exact_neighbors,
                          const int* hnsw_neighbor_indices, int k_neighbors) {

        if (exact_neighbors.empty() || k_neighbors == 0) {
            return 0.0f;
        }

        int intersection = 0;
        std::unordered_set<int> exact_set;

        // Build set of exact neighbor indices
        for (const auto& pair : exact_neighbors) {
            exact_set.insert(pair.second);
        }

        // Count intersection with HNSW neighbors
        for (int i = 0; i < k_neighbors; ++i) {
            if (hnsw_neighbor_indices[i] >= 0 &&
                exact_set.find(hnsw_neighbor_indices[i]) != exact_set.end()) {
                intersection++;
            }
        }

        // Calculate recall as intersection / min(k_neighbors, exact_neighbors.size())
        int min_size = std::min(k_neighbors, (int)exact_neighbors.size());
        return min_size > 0 ? (float)intersection / min_size : 0.0f;
    }

    // FIX19: Build distance matrix with overflow protection for large datasets
    void build_distance_matrix(const float* data, int n_obs, int n_dim, PacMapMetric metric,
                              float* distance_matrix, pacmap_progress_callback_v2 progress_callback,
                              int current_obs, int total_obs) {

        // FIX19: Use int64_t for large dataset support (1M+ points)
        // Calculate total pairs with overflow protection
        if (n_obs <= 1) {
            return;  // No pairs to compute
        }

        bool overflow = false;
        int64_t n_obs_64 = static_cast<int64_t>(n_obs);

        // Safe calculation: n_obs * (n_obs - 1) / 2
        int64_t total_pairs = safe_multiply_int64(n_obs_64, n_obs_64 - 1, &overflow);
        if (overflow || total_pairs < 0) {
            if (progress_callback) {
                progress_callback("ERROR", current_obs, total_obs, 0.0f,
                               "Integer overflow in distance matrix calculation - dataset too large");
            }
            return;
        }
        total_pairs = total_pairs / 2;  // Divide by 2 for upper triangle only

        // Check if total_pairs is reasonable (safety check)
        const int64_t MAX_PAIRS = 1000000000LL;  // 1 billion pairs limit
        if (total_pairs > MAX_PAIRS) {
            if (progress_callback) {
                progress_callback("ERROR", current_obs, total_obs, 0.0f,
                               "Distance matrix too large - consider using approximation methods");
            }
            return;
        }

        int64_t pairs_processed = 0;
        const int64_t REPORT_INTERVAL = 1000;  // Report every 1000 pairs

        // Fill upper triangle of distance matrix with overflow-protected indexing
        for (int i = 0; i < n_obs; ++i) {
            // Check for overflow in matrix indexing: i * n_obs + j
            int64_t row_offset = safe_multiply_int64(static_cast<int64_t>(i), n_obs_64, &overflow);
            if (overflow) {
                if (progress_callback) {
                    progress_callback("ERROR", current_obs + pairs_processed, total_obs, 0.0f,
                                   "Matrix indexing overflow - dataset too large");
                }
                return;
            }

            for (int j = i + 1; j < n_obs; ++j) {
                // Safe matrix indexing
                int64_t index_ij = row_offset + static_cast<int64_t>(j);
                int64_t index_ji = safe_multiply_int64(static_cast<int64_t>(j), n_obs_64, &overflow) + static_cast<int64_t>(i);
                if (overflow) {
                    if (progress_callback) {
                        progress_callback("ERROR", current_obs + pairs_processed, total_obs, 0.0f,
                                       "Matrix indexing overflow - dataset too large");
                    }
                    return;
                }

                // Validate indices are within bounds
                if (index_ij >= static_cast<int64_t>(n_obs * n_obs) ||
                    index_ji >= static_cast<int64_t>(n_obs * n_obs)) {
                    if (progress_callback) {
                        progress_callback("ERROR", current_obs + pairs_processed, total_obs, 0.0f,
                                       "Matrix index out of bounds");
                    }
                    return;
                }

                // Compute distance
                float dist = compute_distance(data + i * n_dim, data + j * n_dim, n_dim, metric);

                // Store in symmetric matrix
                distance_matrix[static_cast<size_t>(index_ij)] = dist;
                distance_matrix[static_cast<size_t>(index_ji)] = dist;

                pairs_processed++;

                // Report progress if callback provided
                if (progress_callback && (pairs_processed % REPORT_INTERVAL == 0 || pairs_processed == total_pairs)) {
                    float percent = static_cast<float>(pairs_processed) / static_cast<float>(total_pairs) * 100.0f;
                    progress_callback("Building Distance Matrix",
                                    current_obs + static_cast<int>(pairs_processed),
                                    total_obs, percent, nullptr);
                }
            }
        }
    }

}