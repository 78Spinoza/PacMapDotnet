#pragma once

#include "pacmap_model.h"
#include <cmath>
#include <algorithm>
#include <unordered_set>

// Distance metric implementations for UMAP
namespace distance_metrics {

    // FIX17.md Step 6: Inlined distance metric functions for performance
    // Note: Using std::sqrt for cross-platform compatibility
    // FIX22 Tier 2: Add const pointers for better compiler optimization
    inline float euclidean_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    inline float cosine_distance(const float* a, const float* b, int dim) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        // Note: Using std::sqrt for cross-platform compatibility
        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        if (norm_a < 1e-10f || norm_b < 1e-10f) return 1.0f;

        float cosine_sim = dot / (norm_a * norm_b);
        cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));

        return 1.0f - cosine_sim;
    }

    inline float manhattan_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dist += std::abs(a[i] - b[i]);
        }
        return dist;
    }

    inline float correlation_distance(const float* a, const float* b, int dim) {
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

        // Note: Using std::sqrt for cross-platform compatibility
        float correlation = num / std::sqrt(den_a * den_b);
        correlation = std::max(-1.0f, std::min(1.0f, correlation));

        return 1.0f - correlation;
    }

    inline float hamming_distance(const float* a, const float* b, int dim) {
        int different = 0;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6f) {
                different++;
            }
        }
        return static_cast<float>(different) / static_cast<float>(dim);
    }

    // FIX17.md Step 6: Inlined distance metric functions (double precision)
    inline double euclidean_distance(const double* a, const double* b, int dim) {
        double dist = 0.0;
        for (int i = 0; i < dim; ++i) {
            double diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    inline double cosine_distance(const double* a, const double* b, int dim) {
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

    inline double manhattan_distance(const double* a, const double* b, int dim) {
        double dist = 0.0;
        for (int i = 0; i < dim; ++i) {
            dist += std::abs(a[i] - b[i]);
        }
        return dist;
    }

    inline double correlation_distance(const double* a, const double* b, int dim) {
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

    inline double hamming_distance(const double* a, const double* b, int dim) {
        int different = 0;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6) {
                different++;
            }
        }
        return static_cast<double>(different) / static_cast<double>(dim);
    }

    // Unified distance computation based on metric type
    // FIX22 Tier 2: Add const pointers for better compiler optimization
    float compute_distance(const float* a, const float* b, int dim, PacMapMetric metric);
    double compute_distance(const double* a, const double* b, int dim, PacMapMetric metric);

    // Data validation for metric-specific requirements
    void validate_metric_data(const float* data, int n_obs, int n_dim, PacMapMetric metric);

    // Zero-norm vector detection for cosine and correlation metrics
    bool detect_zero_norm_vectors(const float* data, int n_obs, int n_dim, PacMapMetric metric);

    // Helper functions for common distance conversion patterns

    // Find k nearest neighbors from a query point to all dataset points (exact search)
    void find_knn_exact(const float* query_point, const float* dataset, int n_obs, int n_dim,
                       PacMapMetric metric, int k_neighbors, std::vector<std::pair<float, int>>& neighbors_out,
                       int query_index = -1);
    void find_knn_exact(const double* query_point, const double* dataset, int n_obs, int n_dim,
                       PacMapMetric metric, int k_neighbors, std::vector<std::pair<float, int>>& neighbors_out,
                       int query_index = -1);

    // Compare two neighbor lists and calculate recall (intersection / union)
    float calculate_recall(const std::vector<std::pair<float, int>>& exact_neighbors,
                          const int* hnsw_neighbor_indices, int k_neighbors);

    // Build distance matrix with progress reporting (all-to-all distances)
    void build_distance_matrix(const float* data, int n_obs, int n_dim, PacMapMetric metric,
                              float* distance_matrix, pacmap_progress_callback_v2 progress_callback = nullptr,
                              int current_obs = 0, int total_obs = 0);

}