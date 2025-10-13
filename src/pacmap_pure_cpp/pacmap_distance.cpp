#include "pacmap_distance.h"
#include <cstdio>  // For fprintf warnings

namespace distance_metrics {

    float euclidean_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    float cosine_distance(const float* a, const float* b, int dim) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        if (norm_a < 1e-10f || norm_b < 1e-10f) return 1.0f;

        float cosine_sim = dot / (norm_a * norm_b);
        cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));

        return 1.0f - cosine_sim;
    }

    float manhattan_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dist += std::abs(a[i] - b[i]);
        }
        return dist;
    }

    float correlation_distance(const float* a, const float* b, int dim) {
        float mean_a = 0.0f, mean_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            mean_a += a[i];
            mean_b += b[i];
        }
        mean_a /= static_cast<float>(dim);
        mean_b /= static_cast<float>(dim);

        float num = 0.0f, den_a = 0.0f, den_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff_a = a[i] - mean_a;
            float diff_b = b[i] - mean_b;
            num += diff_a * diff_b;
            den_a += diff_a * diff_a;
            den_b += diff_b * diff_b;
        }

        if (den_a < 1e-10f || den_b < 1e-10f) return 1.0f;

        float correlation = num / std::sqrt(den_a * den_b);
        correlation = std::max(-1.0f, std::min(1.0f, correlation));

        return 1.0f - correlation;
    }

    float hamming_distance(const float* a, const float* b, int dim) {
        int different = 0;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6f) {
                different++;
            }
        }
        return static_cast<float>(different) / static_cast<float>(dim);
    }

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

    // Double precision distance functions
    double euclidean_distance(const double* a, const double* b, int dim) {
        double dist = 0.0;
        for (int i = 0; i < dim; ++i) {
            double diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    double cosine_distance(const double* a, const double* b, int dim) {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;

        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        if (norm_a < 1e-10 || norm_b < 1e-10) return 1.0;

        double cosine_sim = dot / (norm_a * norm_b);
        cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));

        return 1.0 - cosine_sim;
    }

    double manhattan_distance(const double* a, const double* b, int dim) {
        double dist = 0.0;
        for (int i = 0; i < dim; ++i) {
            dist += std::abs(a[i] - b[i]);
        }
        return dist;
    }

    double correlation_distance(const double* a, const double* b, int dim) {
        double mean_a = 0.0, mean_b = 0.0;
        for (int i = 0; i < dim; ++i) {
            mean_a += a[i];
            mean_b += b[i];
        }
        mean_a /= static_cast<double>(dim);
        mean_b /= static_cast<double>(dim);

        double num = 0.0, den_a = 0.0, den_b = 0.0;
        for (int i = 0; i < dim; ++i) {
            double diff_a = a[i] - mean_a;
            double diff_b = b[i] - mean_b;
            num += diff_a * diff_b;
            den_a += diff_a * diff_a;
            den_b += diff_b * diff_b;
        }

        if (den_a < 1e-10 || den_b < 1e-10) return 1.0;

        double correlation = num / std::sqrt(den_a * den_b);
        correlation = std::max(-1.0, std::min(1.0, correlation));

        return 1.0 - correlation;
    }

    double hamming_distance(const double* a, const double* b, int dim) {
        int different = 0;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6) {
                different++;
            }
        }
        return static_cast<double>(different) / static_cast<double>(dim);
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

    // Build distance matrix with progress reporting (all-to-all distances)
    void build_distance_matrix(const float* data, int n_obs, int n_dim, PacMapMetric metric,
                              float* distance_matrix, pacmap_progress_callback_v2 progress_callback,
                              int current_obs, int total_obs) {

        int total_pairs = n_obs * (n_obs - 1) / 2;
        int pairs_processed = 0;

        // Fill upper triangle of distance matrix
        for (int i = 0; i < n_obs; ++i) {
            for (int j = i + 1; j < n_obs; ++j) {
                float dist = compute_distance(data + i * n_dim, data + j * n_dim, n_dim, metric);
                distance_matrix[i * n_obs + j] = dist;
                distance_matrix[j * n_obs + i] = dist;  // Symmetric

                pairs_processed++;

                // Report progress if callback provided
                if (progress_callback && (pairs_processed % 1000 == 0 || pairs_processed == total_pairs)) {
                    float percent = (float)pairs_processed / total_pairs * 100.0f;
                    progress_callback("Building Distance Matrix", current_obs + pairs_processed, total_obs, percent, nullptr);
                }
            }
        }
    }

}