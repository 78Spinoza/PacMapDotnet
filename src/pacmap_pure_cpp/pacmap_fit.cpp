#include "pacmap_fit.h"
#include "pacmap_triplet_sampling.h"
#include "pacmap_optimization.h"
#include "pacmap_simple_wrapper.h"
#include "pacmap_distance.h"
#include "pacmap_system_info.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <chrono>
#include <cmath>

namespace fit_utils {

    // Validate n_neighbors parameter and issue warning for adaptive formula recommendation
    void validate_n_neighbors_parameter(int n_neighbors, int n_samples, pacmap_progress_callback_internal progress_callback) {
        if (n_samples >= 10000) {
            // Calculate recommended n_neighbors using adaptive formula
            // n_neighbors = 10 + 15 * (log10(n) - 4)
            double log10_n = std::log10(static_cast<double>(n_samples));
            int recommended_neighbors = static_cast<int>(10 + 15 * (log10_n - 4.0));

            // Clamp to reasonable bounds
            recommended_neighbors = std::max(10, std::min(100, recommended_neighbors));

            if (n_neighbors != recommended_neighbors) {
                std::string warning_msg = "Parameter Warning: n_neighbors=" + std::to_string(n_neighbors) +
                                       " is not optimal for " + std::to_string(n_samples) + " samples. " +
                                       "Recommended: n_neighbors=" + std::to_string(recommended_neighbors) +
                                       " (using formula: 10 + 15 * (log10(" + std::to_string(n_samples) + ") - 4) = " +
                                       std::to_string(recommended_neighbors) + ")";

                if (progress_callback) {
                    progress_callback("Parameter Warning", 0, 1, 0.0f, warning_msg.c_str());
                }
            }
        }
        else if (n_samples < 10000 && n_neighbors != 10) {
            std::string warning_msg = std::string("Parameter Warning: For small datasets (<10,000 samples), recommended n_neighbors=10, ") +
                                   "but you used n_neighbors=" + std::to_string(n_neighbors);

            if (progress_callback) {
                progress_callback("Parameter Warning", 0, 1, 0.0f, warning_msg.c_str());
            }
        }
    }

    // Validate MN_ratio and FP_ratio parameters and issue warning for incorrect relationship
    void validate_ratio_parameters(float mn_ratio, float fp_ratio, pacmap_progress_callback_internal progress_callback) {
        // Expected relationship: FP_ratio = 4 * MN_ratio (with defaults MN_ratio=0.5, FP_ratio=2.0)
        float expected_fp_ratio = 4.0f * mn_ratio;
        float tolerance = 0.01f; // Small tolerance for floating point comparison

        if (std::abs(fp_ratio - expected_fp_ratio) > tolerance) {
            std::string warning_msg = "Parameter Warning: MN_ratio=" + std::to_string(mn_ratio) +
                                   " and FP_ratio=" + std::to_string(fp_ratio) +
                                   " don't follow recommended relationship. " +
                                   "Expected: FP_ratio=4*MN_ratio (FP_ratio=" +
                                   std::to_string(expected_fp_ratio) + " for MN_ratio=" +
                                   std::to_string(mn_ratio) + "). " +
                                   "Default recommendation: MN_ratio=0.5, FP_ratio=2.0";

            if (progress_callback) {
                progress_callback("Parameter Warning", 0, 1, 0.0f, warning_msg.c_str());
            }
        }

        // Additional warnings for extreme values
        if (mn_ratio < 0.1f || mn_ratio > 2.0f) {
            std::string warning_msg = "Parameter Warning: MN_ratio=" + std::to_string(mn_ratio) +
                                   " is outside recommended range [0.1, 2.0]. " +
                                   "Values outside this range may cause unstable optimization.";

            if (progress_callback) {
                progress_callback("Parameter Warning", 0, 1, 0.0f, warning_msg.c_str());
            }
        }

        if (fp_ratio < 0.5f || fp_ratio > 5.0f) {
            std::string warning_msg = "Parameter Warning: FP_ratio=" + std::to_string(fp_ratio) +
                                   " is outside recommended range [0.5, 5.0]. " +
                                   "Values outside this range may affect global structure preservation.";

            if (progress_callback) {
                progress_callback("Parameter Warning", 0, 1, 0.0f, warning_msg.c_str());
            }
        }
    }

    // Auto-tune HNSW parameters using subsample (ERROR12 Priority 1)
    // Tests different M/ef_construction combinations to find optimal balance of speed and recall
    int autoTuneHNSWParams(PacMapModel* model, const double* data, int n_obs, int n_dim,
                           pacmap_progress_callback_internal callback) {
        // Cap subsample at 30,000 points for efficiency (ERROR12 recommendation)
        int subsample_size = std::min(30000, n_obs / 10);
        subsample_size = std::max(1000, subsample_size); // Minimum 1000 for reliability

        if (callback) {
            std::string msg = "Sampling " + std::to_string(subsample_size) + " points for HNSW tuning";
            callback("HNSW Auto-Tuning", 0, 100, 0.0f, msg.c_str());
        }

        // Sample random points
        std::vector<float> subsample(static_cast<size_t>(subsample_size) * static_cast<size_t>(n_dim));
        std::mt19937 rng(model->random_seed >= 0 ? model->random_seed : 42);
        std::uniform_int_distribution<int> dist(0, n_obs - 1);

        for (int i = 0; i < subsample_size; i++) {
            int idx = dist(rng);
            std::memcpy(&subsample[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                       data + static_cast<size_t>(idx) * static_cast<size_t>(n_dim),
                       n_dim * sizeof(float));
        }

        // Test candidates: {M, ef_construction} pairs optimized for speed & memory
        std::vector<std::tuple<int, int>> candidates = {
            {16, 100},  // Fastest, lowest memory
            {16, 150},  // Balanced (recommended default)
            {24, 150},  // Better recall, moderate memory
            {32, 200}   // Best quality, higher memory
        };

        // For each candidate, use higher search ef to achieve 95%+ recall
        // Search ef should typically be >= ef_construction for good recall
        std::vector<std::tuple<int, int, int>> test_candidates = {
            {16, 100, 200},  // M=16, ef_construction=100, ef_search=200
            {16, 150, 300},  // M=16, ef_construction=150, ef_search=300
            {24, 150, 300},  // M=24, ef_construction=150, ef_search=300
            {32, 200, 400}   // M=32, ef_construction=200, ef_search=400
        };

        float best_recall = 0.0f;
        int best_M = 16, best_ef = 150;
        double best_time = 1e9;
        bool tuning_success = false;

        for (size_t i = 0; i < test_candidates.size(); i++) {
            auto [M, ef_construction, ef_search] = test_candidates[i];

            try {
                auto start = std::chrono::steady_clock::now();

                // Create HNSW index with candidate parameters
                std::unique_ptr<hnswlib::L2Space> space = std::make_unique<hnswlib::L2Space>(n_dim);
                std::unique_ptr<hnswlib::HierarchicalNSW<float>> index =
                    std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), subsample_size, M, ef_construction);

                // Set search ef parameter for high recall testing
                // Use higher search ef to achieve 95%+ recall
                index->setEf(ef_search);

                // Build index
                for (int j = 0; j < subsample_size; j++) {
                    index->addPoint(&subsample[static_cast<size_t>(j) * static_cast<size_t>(n_dim)], j);
                }

                auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();

                // Test recall on 100 random queries
                float avg_recall = 0.0f;
                auto query_start = std::chrono::steady_clock::now();

                for (int q = 0; q < 100; q++) {
                    int query_idx = q % subsample_size;

                    // Exact KNN
                    std::vector<std::pair<float, int>> exact_knn;
                    distance_metrics::find_knn_exact(
                        &subsample[static_cast<size_t>(query_idx) * static_cast<size_t>(n_dim)],
                        subsample.data(),
                        subsample_size, n_dim, model->metric,
                        model->n_neighbors, exact_knn, query_idx
                    );

                    // HNSW KNN
                    auto hnsw_res = index->searchKnn(&subsample[static_cast<size_t>(query_idx) * static_cast<size_t>(n_dim)], model->n_neighbors);
                    std::vector<int> hnsw_indices;
                    while (!hnsw_res.empty()) {
                        hnsw_indices.push_back(static_cast<int>(hnsw_res.top().second));
                        hnsw_res.pop();
                    }

                    // Calculate recall
                    avg_recall += distance_metrics::calculate_recall(exact_knn, hnsw_indices.data(), model->n_neighbors);
                }
                avg_recall /= 100.0f;

                auto query_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - query_start).count();

                double total_time = static_cast<double>(build_time + query_time);

                // Select best: require >95% recall, minimize time
                if (avg_recall >= 0.95f && total_time < best_time) {
                    best_recall = avg_recall;
                    best_time = total_time;
                    best_M = M;
                    best_ef = ef_construction; // Store construction ef for final model
                    tuning_success = true;
                }

                if (callback) {
                    std::string msg = "M=" + std::to_string(M) + ", ef_c=" + std::to_string(ef_construction) +
                                      ", ef_s=" + std::to_string(ef_search) + ", recall=" + std::to_string(static_cast<int>(avg_recall * 100)) + "%";
                    callback("HNSW Tuning", static_cast<int>((i + 1) * 25), 100, static_cast<float>((i + 1) * 25), msg.c_str());
                }

            } catch (const std::exception& e) {
                // Candidate failed, continue to next
                if (callback) {
                    std::string msg = "M=" + std::to_string(M) + ", ef_c=" + std::to_string(ef_construction) +
                                      ", ef_s=" + std::to_string(ef_search) + " failed";
                    callback("Warning", static_cast<int>((i + 1) * 25), 100, static_cast<float>((i + 1) * 25), msg.c_str());
                }
                continue;
            }
        }

        // Use defaults if tuning failed
        if (!tuning_success) {
            if (callback) {
                callback("Warning", 100, 100, 100.0f, "HNSW tuning failed, using defaults M=16, ef=150");
            }
            best_M = 16;
            best_ef = 150;
        }

        // Apply tuned parameters
        model->hnsw_m = best_M;
        model->hnsw_ef_construction = best_ef;
        model->hnsw_ef_search = std::max(best_ef * 2, 200); // Use higher search ef for production

        if (callback) {
            std::string msg = "Selected M=" + std::to_string(best_M) + ", ef_c=" + std::to_string(best_ef) +
                              ", ef_s=" + std::to_string(model->hnsw_ef_search) +
                              ", recall=" + std::to_string(static_cast<int>(best_recall * 100)) + "%";
            callback("HNSW Auto-Tuning Complete", 100, 100, 100.0f, msg.c_str());
        }

        return PACMAP_SUCCESS;
    }

    // Main PACMAP fitting function with proper PACMAP workflow (internal implementation)
    int internal_pacmap_fit_with_progress_v2(PacMapModel* model,
        double* data,
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
        double* embedding,
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


        // Report system capabilities and performance optimizations
        if (progress_callback) {
            pacmap_system::report_system_capabilities(progress_callback);
        } else {
        }

        if (embedding_dim > 50) {
            if (progress_callback) {
                progress_callback("Error", 0, 1, 0.0f, "Embedding dimension must be <= 50");
            }
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        // Validate n_neighbors parameter and issue adaptive formula warning
        
        validate_n_neighbors_parameter(n_neighbors, n_obs, progress_callback);
        

        // Validate MN_ratio and FP_ratio parameters and issue relationship warning
        
        validate_ratio_parameters(mn_ratio, fp_ratio, progress_callback);
        

        
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
            

              model->phase1_iters = phase1_iters;
            model->phase2_iters = phase2_iters;
            model->phase3_iters = phase3_iters;
            model->metric = metric;
            model->random_seed = random_seed;
            model->use_quantization = (use_quantization != 0);
            model->force_exact_knn = (force_exact_knn != 0); // Convert int to bool
            model->hnsw_m = M > 0 ? M : 16;
            model->hnsw_ef_construction = ef_construction > 0 ? ef_construction : 150;  // ERROR12: Lowered from 200 for faster index building
            model->hnsw_ef_search = ef_search > 0 ? ef_search : 100;  // ERROR12: Lowered from 200 for 2x faster queries
            

            // Auto-tune HNSW parameters if requested (ERROR12 Priority 1)
            // Only run for HNSW mode (not force_exact_knn) and datasets >= 5000 points
            
            if (autoHNSWParam && !force_exact_knn && n_obs >= 5000) {
                
                int tune_result = autoTuneHNSWParams(model, data, n_obs, n_dim, progress_callback);
                
                if (tune_result != PACMAP_SUCCESS) {
                    // Tuning failed, but continue with defaults
                    if (progress_callback) {
                        
                        progress_callback("Warning", 0, 1, 0.0f, "HNSW auto-tuning failed, using defaults M=16, ef=150");
                    }
                }
            }

            
            if (progress_callback) {
                
                progress_callback("Initializing PACMAP", 1, 10, 10.0f, "Setting up model parameters");
            } else {
                
            }

            // Convert input data to vector format and store in model for KNN direct mode
            // Convert double input to float for internal storage (API boundary conversion)
            
            std::vector<double> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));
            
            model->training_data = input_data; // Store training data for KNN direct mode persistence
            

            // CRITICAL FIX v2.8.4: Match Python preprocessing exactly - min-max + mean centering
            
            // Python reference (pacmap.py lines 371-375):
            //   xmin, xmax = (np.min(X), np.max(X))
            //   X -= xmin          # Min offset
            //   X /= xmax          # Scale to [0,1]
            //   xmean = np.mean(X, axis=0)
            //   X -= xmean         # Mean centering AFTER scaling
            //
            // Previous C++ implementation used z-score normalization which creates different data scaling
            // This affected distance calculations and force dynamics, contributing to oval formation

            // Step 1: Find global min and max values across entire dataset - double precision
            
            double xmin = std::numeric_limits<double>::max();
            double xmax = std::numeric_limits<double>::lowest();
            
            for (int i = 0; i < n_obs; i++) {
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    xmin = std::min(xmin, input_data[idx]);
                    xmax = std::max(xmax, input_data[idx]);
                }
            }
            

            // Step 2: Apply min-max scaling to [0,1] range - double precision
            
            std::vector<double> normalized_data = input_data;
            for (int i = 0; i < n_obs; i++) {
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    normalized_data[idx] = (input_data[idx] - xmin) / xmax;  // Scale to [0,1]
                }
            }
            

            // Step 3: Calculate column-wise means AFTER scaling (matching Python exactly) - double precision
            
            model->feature_means.resize(n_dim, 0.0);
            model->feature_stds.resize(n_dim, 1.0);  // Keep for compatibility but not used in new normalization

            for (int j = 0; j < n_dim; j++) {
                double sum = 0.0;
                for (int i = 0; i < n_obs; i++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    sum += normalized_data[idx];  // Use scaled data for mean calculation
                }
                model->feature_means[j] = sum / static_cast<double>(n_obs);
            }
            

            // Step 4: Apply mean centering AFTER min-max scaling (matching Python exactly) - double precision
            
            for (int i = 0; i < n_obs; i++) {
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    normalized_data[idx] -= model->feature_means[j];  // Mean centering
                }
            }
            

            // Store preprocessing parameters for transform consistency - double precision
            model->xmin = xmin;
            model->xmax = xmax;


            if (progress_callback) {
                progress_callback("Data Normalization", 2, 10, 20.0f, "Computing feature statistics");
            }

            // PACMAP Step 1: Triplet Sampling
            
            if (progress_callback) {
                
                progress_callback("Triplet Sampling", 3, 10, 30.0f, "Sampling neighbor, mid-near, and far triplets");
            } else {
                
            }

            
            sample_triplets(model, normalized_data.data(), progress_callback);
            

            // ERROR12 Priority 12: Validate triplets before optimization
            if (model->triplets.empty()) {
                if (progress_callback) {
                    progress_callback("Error", 0, 1, 0.0f, "No triplets generated for optimization - check your parameters");
                }
                return PACMAP_ERROR_INVALID_PARAMS;
            }


            // Initialize Adam optimizer state (handled in optimization loop)
            size_t embedding_size = static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim);
            // Adam state is now initialized in the optimization function


            if (progress_callback) {
                progress_callback("Optimizer Setup", 4, 10, 40.0f, "Initializing Adam optimizer state");
            }

            // CRITICAL FIX: Initialize embedding to match Rust implementation exactly
            // Rust uses simple random normal initialization without post-scaling
            std::mt19937 generator(random_seed >= 0 ? random_seed : 42);
            std::normal_distribution<float> dist(0.0f, 1e-4f); // CRITICAL FIX: Match reference implementation exactly
            for (size_t i = 0; i < embedding_size; i++) {
                embedding[i] = dist(generator);
            }

            // CRITICAL FIX: Removed scaling to std=1e-4f to match Rust exactly
            // Rust PACMAP does not apply variance scaling after initialization
            // Let Adam optimizer handle the natural scale of the embedding

            // PACMAP Step 2: Three-phase Optimization
            if (progress_callback) {
                progress_callback("Three-phase Optimization", 5, 10, 50.0f, "Starting three-phase PACMAP optimization");
            }

            auto opt_start = std::chrono::high_resolution_clock::now();

            optimize_embedding(model, embedding, progress_callback);

            auto opt_end = std::chrono::high_resolution_clock::now();
            auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start);

            // Debug final embedding statistics
            std::vector<float> embedding_vec(n_obs * embedding_dim);
            for (int i = 0; i < n_obs * embedding_dim; ++i) {
                embedding_vec[i] = static_cast<float>(embedding[i]);
            }
            float init_mean = std::accumulate(embedding_vec.begin(), embedding_vec.end(), 0.0f) / embedding_vec.size();
            float init_std = 0.0f;
            for (float e : embedding_vec) init_std += (e - init_mean) * (e - init_mean);
            init_std = std::sqrt(init_std / embedding_vec.size());
      
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
                progress_callback("Building Embedding Space Index", 9, 10, 90.0f, "Creating HNSW index for AI inference");
            }

            try {
                // Create persistent L2Space owned by the model (fixes AccessViolationException)
                model->embedding_space = std::make_unique<hnswlib::L2Space>(model->n_components);

                // Construct embedding space HNSW index with persistent metric space
                model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                    model->embedding_space.get(),
                    model->n_samples,
                    model->hnsw_m,
                    model->hnsw_ef_construction,
                    model->random_seed
                );
                model->embedding_space_index->setEf(model->hnsw_ef_search);

                // Add all embedding points to the embedding space HNSW index
                // Convert double to float for HNSW (which only accepts float*)
                std::vector<float> embedding_float(model->embedding.size());
                for (size_t i = 0; i < model->embedding.size(); ++i) {
                    embedding_float[i] = static_cast<float>(model->embedding[i]);
                }
                for (int i = 0; i < model->n_samples; i++) {
                    const float* embedding_point = &embedding_float[static_cast<size_t>(i) * static_cast<size_t>(model->n_components)];
                    model->embedding_space_index->addPoint(embedding_point, static_cast<size_t>(i));
                }

                if (progress_callback) {
                    progress_callback("Embedding Space Index Complete", 10, 10, 100.0f,
                                    "HNSW embedding index built successfully for AI inference");
                }
            }
            catch (const std::exception& e) {
                // Embedding space index creation failed - not critical for basic functionality
                model->embedding_space_index = nullptr;
                model->embedding_space = nullptr;
                if (progress_callback) {
                    progress_callback("Warning", 100, 100, 100.0f,
                                    ("Embedding space HNSW index creation failed: " + std::string(e.what())).c_str());
                }
                // Continue without embedding space index - transform will still work for basic projection
            }

            model->is_fitted = true;

            if (progress_callback) {
                progress_callback("PACMAP Complete", 10, 10, 100.0f, "PACMAP fitting completed successfully");
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

