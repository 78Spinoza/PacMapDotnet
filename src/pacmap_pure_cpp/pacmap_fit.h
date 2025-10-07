#pragma once

#include "pacmap_model.h"
#include "pacmap_hnsw_utils.h"
#include "pacmap_progress_utils.h"
#include "pacmap_distance.h"
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <atomic>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fit_utils {

    // Import distance metrics from dedicated module
    using namespace distance_metrics;

    // Core k-NN graph building with HNSW optimization
    void build_knn_graph(
        const std::vector<float>& data,
        int n_obs, int n_dim, int n_neighbors,
        PacMapMetric metric, PacMapModel* model,
        std::vector<int>& nn_indices,
        std::vector<double>& nn_distances,
        int force_exact_knn,
        pacmap_progress_callback_v2 progress_callback = nullptr,
        int autoHNSWParam = 1
    );

    // Convert uwot smooth k-NN output to edge list format
    void convert_to_edges(
        const std::vector<int>& nn_indices,
        const std::vector<double>& nn_weights,
        int n_obs, int n_neighbors,
        std::vector<unsigned int>& heads,
        std::vector<unsigned int>& tails,
        std::vector<double>& weights
    );

    // Calculate UMAP parameters from spread and min_dist
    void calculate_ab_from_spread_and_min_dist(PacMapModel* model);

    // Compute normalization parameters for training data
    void compute_normalization(
        const std::vector<float>& data,
        int n_obs, int n_dim,
        std::vector<float>& feature_means,
        std::vector<float>& feature_stds
    );

    // Compute neighbor statistics for safety analysis
    void compute_neighbor_statistics(PacMapModel* model, const std::vector<float>& normalized_data);

    // Main fit function with progress reporting
    int uwot_fit_with_progress(
        PacMapModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        float spread,
        int n_epochs,
        PacMapMetric metric,
        float* embedding,
        pacmap_progress_callback_v2 progress_callback,
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search,
        int use_quantization,
        int random_seed = -1,
        int autoHNSWParam = 1
    );

    // Enhanced v2 function with loss reporting
    int uwot_fit_with_progress_v2(
        PacMapModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        float spread,
        int n_epochs,
        PacMapMetric metric,
        float* embedding,
        pacmap_progress_callback_v2 progress_callback,
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search,
        int use_quantization = 0,
        int random_seed = -1,
        int autoHNSWParam = 1
    );

    // Main PACMAP fitting function with proper PACMAP workflow (internal implementation)
    int internal_pacmap_fit_with_progress_v2(
        PacMapModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float mn_ratio,
        float fp_ratio,
        float learning_rate,
        int n_iters,
        int phase1_iters,
        int phase2_iters,
        int phase3_iters,
        PacMapMetric metric,
        float* embedding,
        pacmap_progress_callback_v2 progress_callback,
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search,
        int use_quantization,
        int random_seed,
        int autoHNSWParam
    );

    // Helper functions for uwot_fit refactoring
    namespace fit_helpers {
        // HNSW recall validation and auto-tuning
        bool validate_hnsw_recall(PacMapModel* model, const float* data, int n_obs, int n_dim,
                                  int n_neighbors, PacMapMetric metric, pacmap_progress_callback_v2 progress_callback);

        // Auto-tune ef_search parameter based on recall measurement
        bool auto_tune_ef_search(PacMapModel* model, const float* data, int n_obs, int n_dim,
                                 int n_neighbors, PacMapMetric metric, pacmap_progress_callback_v2 progress_callback);

        // Initialize random number generators with seed
        void initialize_random_generators(PacMapModel* model);
    }
}