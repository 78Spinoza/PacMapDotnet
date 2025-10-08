#include "pacmap_fit.h"
#include "pacmap_triplet_sampling.h"
#include "pacmap_optimization.h"
#include "pacmap_simple_wrapper.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <chrono>

namespace fit_utils {

    // Main PACMAP fitting function with proper PACMAP workflow (internal implementation)
    int internal_pacmap_fit_with_progress_v2(PacMapModel* model,
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
        pacmap_progress_callback_internal progress_callback,
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search,
        int use_quantization,
        int random_seed,
        int autoHNSWParam,
        float initialization_std_dev) {

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 ||
            embedding_dim <= 0 || n_neighbors <= 0 || n_neighbors >= n_obs) {
            if (progress_callback) {
                progress_callback("Error", 0, 1, 0.0f, "Invalid parameters");
            }
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        if (embedding_dim > 50) {
            if (progress_callback) {
                progress_callback("Error", 0, 1, 0.0f, "Embedding dimension must be <= 50");
            }
            return PACMAP_ERROR_INVALID_PARAMS;
        }

              try {
            // Initialize PACMAP model parameters
            model->n_samples = n_obs;
            model->n_features = n_dim;
            model->n_components = embedding_dim;
            model->n_neighbors = n_neighbors;
            model->mn_ratio = mn_ratio;
            model->fp_ratio = fp_ratio;
            model->learning_rate = learning_rate;
            model->initialization_std_dev = initialization_std_dev;

            printf("[FIT DEBUG] Init: n_samples=%d, n_dim=%d, embed_dim=%d, n_neighbors=%d, mn_ratio=%.2f, fp_ratio=%.2f, init_std_dev=%.6f\n",
                   n_obs, n_dim, embedding_dim, n_neighbors, mn_ratio, fp_ratio, initialization_std_dev);
            model->phase1_iters = phase1_iters;
            model->phase2_iters = phase2_iters;
            model->phase3_iters = phase3_iters;
            model->metric = metric;
            model->random_seed = random_seed;
            model->use_quantization = (use_quantization != 0);
            model->force_exact_knn = (force_exact_knn != 0); // Convert int to bool
            model->hnsw_m = M > 0 ? M : 16;
            model->hnsw_ef_construction = ef_construction > 0 ? ef_construction : 200;
            model->hnsw_ef_search = ef_search > 0 ? ef_search : 200;

            
            if (progress_callback) {
                progress_callback("Initializing PACMAP", 0, 100, 5.0f, "Setting up model parameters");
            }

            // Convert input data to vector format and store in model for KNN direct mode
            std::vector<float> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));
            model->training_data = input_data; // Store training data for KNN direct mode persistence

            // Compute normalization parameters
            model->feature_means.resize(n_dim, 0.0f);
            model->feature_stds.resize(n_dim, 1.0f);

            // Calculate feature means
            for (int j = 0; j < n_dim; j++) {
                float sum = 0.0f;
                for (int i = 0; i < n_obs; i++) {
                    sum += input_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)];
                }
                model->feature_means[j] = sum / static_cast<float>(n_obs);
            }

            // Calculate feature standard deviations
            for (int j = 0; j < n_dim; j++) {
                float sum_sq = 0.0f;
                for (int i = 0; i < n_obs; i++) {
                    float diff = input_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)] - model->feature_means[j];
                    sum_sq += diff * diff;
                }
                model->feature_stds[j] = std::sqrt(sum_sq / static_cast<float>(n_obs - 1));
                if (model->feature_stds[j] < 1e-8f) model->feature_stds[j] = 1.0f; // Prevent division by zero
            }

            // Normalize data
            std::vector<float> normalized_data = input_data;
            for (int i = 0; i < n_obs; i++) {
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    normalized_data[idx] = (input_data[idx] - model->feature_means[j]) / model->feature_stds[j];
                }
            }


            if (progress_callback) {
                progress_callback("Data Normalization", 1, 100, 10.0f, "Computing feature statistics");
            }

            // PACMAP Step 1: Triplet Sampling
            if (progress_callback) {
                progress_callback("Triplet Sampling", 2, 100, 15.0f, "Sampling neighbor, mid-near, and far triplets");
            }

            sample_triplets(model, normalized_data.data(), progress_callback);

            printf("[FIT DEBUG] Triplets: total=%d, neighbor=%d, mn=%d, fp=%d\n",
                   model->total_triplets, model->neighbor_triplets, model->mid_near_triplets, model->far_triplets);

            // Initialize Adam optimizer state (handled in optimization loop)
            size_t embedding_size = static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim);
            // Adam state is now initialized in the optimization function


            if (progress_callback) {
                progress_callback("Optimizer Setup", 3, 100, 20.0f, "Initializing Adam optimizer state");
            }

            // CRITICAL FIX: Initialize embedding to match Rust implementation exactly
            // Rust uses simple random normal initialization without post-scaling
            std::mt19937 generator(random_seed >= 0 ? random_seed : 42);
            std::normal_distribution<float> dist(0.0f, 0.1f); // Standard initialization similar to Rust
            for (size_t i = 0; i < embedding_size; i++) {
                embedding[i] = dist(generator);
            }

            // CRITICAL FIX: Removed scaling to std=1e-4f to match Rust exactly
            // Rust PACMAP does not apply variance scaling after initialization
            // Let Adam optimizer handle the natural scale of the embedding

            // PACMAP Step 2: Three-phase Optimization
            if (progress_callback) {
                progress_callback("Optimization", 4, 100, 25.0f, "Starting three-phase PACMAP optimization");
            }

            auto opt_start = std::chrono::high_resolution_clock::now();

            optimize_embedding(model, embedding, progress_callback);

            auto opt_end = std::chrono::high_resolution_clock::now();
            auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start);

            // Debug final embedding statistics
            std::vector<float> embedding_vec(embedding, embedding + n_obs * embedding_dim);
            float init_mean = std::accumulate(embedding_vec.begin(), embedding_vec.end(), 0.0f) / embedding_vec.size();
            float init_std = 0.0f;
            for (float e : embedding_vec) init_std += (e - init_mean) * (e - init_mean);
            init_std = std::sqrt(init_std / embedding_vec.size());
            printf("[FIT DEBUG] Final Embedding: mean=%.4f, std=%.4f, target_std=%.6f\n",
                   init_mean, init_std, initialization_std_dev);

            // Compute embedding statistics for transform safety
            std::vector<float> embedding_distances;
            int sample_size = std::min(n_obs, 1000);
            for (int i = 0; i < sample_size; i++) {
                for (int j = i + 1; j < sample_size; j++) {
                    float dist = 0.0f;
                    for (int d = 0; d < embedding_dim; d++) {
                        float diff = embedding[static_cast<size_t>(i) * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                                    embedding[static_cast<size_t>(j) * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                        dist += diff * diff;
                    }
                    embedding_distances.push_back(std::sqrt(dist));
                }
            }

            if (!embedding_distances.empty()) {
                std::sort(embedding_distances.begin(), embedding_distances.end());
                model->min_embedding_distance = embedding_distances.front();
                model->p95_embedding_distance = embedding_distances[static_cast<size_t>(0.95 * embedding_distances.size())];
                model->p99_embedding_distance = embedding_distances[static_cast<size_t>(0.99 * embedding_distances.size())];
                model->mean_embedding_distance = std::accumulate(embedding_distances.begin(), embedding_distances.end(), 0.0f) / embedding_distances.size();

                float variance = 0.0f;
                for (float dist : embedding_distances) {
                    float diff = dist - model->mean_embedding_distance;
                    variance += diff * diff;
                }
                model->std_embedding_distance = std::sqrt(variance / embedding_distances.size());

                // Outlier thresholds
                model->mild_embedding_outlier_threshold = model->mean_embedding_distance + 2.5f * model->std_embedding_distance;
                model->extreme_embedding_outlier_threshold = model->mean_embedding_distance + 4.0f * model->std_embedding_distance;
            }

    
            // Build embedding space HNSW index for AI inference and transform analysis
            if (progress_callback) {
                progress_callback("Building Embedding Space Index", 99, 100, 99.0f, "Creating HNSW index for AI inference");
            }

            try {
                // Create embedding space HNSW index using the fitted embedding coordinates
                // This enables the second step of transform: searching in embedding space
                static hnswlib::L2Space embedding_space(model->n_components);
                model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                    &embedding_space,
                    model->n_samples,
                    model->hnsw_m,
                    model->hnsw_ef_construction
                );
                model->embedding_space_index->setEf(model->hnsw_ef_search);

                // Add all embedding points to the embedding space HNSW index
                for (int i = 0; i < model->n_samples; i++) {
                    const float* embedding_point = &model->embedding[static_cast<size_t>(i) * static_cast<size_t>(model->n_components)];
                    model->embedding_space_index->addPoint(embedding_point, static_cast<size_t>(i));
                }

                if (progress_callback) {
                    progress_callback("Embedding Space Index Complete", 100, 100, 100.0f,
                                    "HNSW embedding index built successfully for AI inference");
                }
            }
            catch (const std::exception& e) {
                // Embedding space index creation failed - not critical for basic functionality
                if (progress_callback) {
                    progress_callback("Warning", 100, 100, 100.0f,
                                    ("Embedding space HNSW index creation failed: " + std::string(e.what())).c_str());
                }
                // Continue without embedding space index - transform will still work for basic projection
            }

            model->is_fitted = true;

            if (progress_callback) {
                progress_callback("PACMAP Complete", 100, 100, 100.0f, "PACMAP fitting completed successfully");
            }

            return PACMAP_SUCCESS;
        }
        catch (const std::exception& e) {
            if (progress_callback) {
                progress_callback("Error", 0, 1, 0.0f, e.what());
            }
            return PACMAP_ERROR_MEMORY;
        }
    }

} // namespace fit_utils

