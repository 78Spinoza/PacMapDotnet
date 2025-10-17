#include "pacmap_optimization.h"
#include "pacmap_gradient.h"
#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include "pacmap_simd_utils.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <chrono>

// ERROR14 Step 3: Eigen SIMD vectorization for Adam optimizer
#include <Eigen/Dense>

void optimize_embedding(PacMapModel* model, double* embedding_out, pacmap_progress_callback_internal callback) {

    // Python hybrid approach: Use double precision internally for numerical stability
    std::vector<double> embedding(model->n_samples * model->n_components);

    // Initialize embedding with proper random normal distribution using model's initialization_std_dev parameter
    // Python reference: np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    if (callback) {
        char init_msg[256];
        snprintf(init_msg, sizeof(init_msg), "INITIALIZATION: Using std_dev=%.6f for %jd samples x %jd components",
                model->initialization_std_dev, (intmax_t)model->n_samples, (intmax_t)model->n_components);
        callback("Initialization", 0, 0, 0.0f, init_msg);
    }
    initialize_random_embedding_double(embedding, model->n_samples, model->n_components, model->rng, model->initialization_std_dev);

    
    int total_iters = model->phase1_iters + model->phase2_iters + model->phase3_iters;

    if (model->triplets_flat.empty()) {
        if (callback) callback("Error", 0, 100, 0.0f, "No triplets available for optimization");
        return;
    }

    // Initialize Adam optimizer state - use double precision for gradients to match embedding precision
    std::vector<double> gradients(embedding.size());

    // Loss history for convergence monitoring
    std::vector<float> loss_history;
    loss_history.reserve(total_iters);

    // Initialize Adam optimizer state in model - store as double precision internally
    model->adam_m.assign(embedding.size(), 0.0);
    model->adam_v.assign(embedding.size(), 0.0);

                
    if (callback) callback("Starting Optimization", 0, 0, 0.0f, nullptr);

    auto loop_start = std::chrono::high_resolution_clock::now();

    // Main optimization loop with three phases
    for (int iter = 0; iter < total_iters; ++iter) {
        // FIX21: Get three-phase weights using iteration-based boundaries (matching Python exactly)
        auto [w_n, w_mn, w_f] = get_weights(iter, model->phase1_iters, model->phase2_iters);

        // MEMORY FIX: Compute gradients using flat triplet storage to prevent allocation failures
        compute_gradients_flat(embedding, model->triplets_flat, gradients,
                             w_n, w_mn, w_f, model->n_components, callback);

        // ERROR13 FIX: Choose optimizer based on Python reference vs Adam optimization
        // Python reference uses simple SGD: embedding -= learning_rate * gradients
        // Our implementation defaults to Adam but allows SGD for exact Python matching

        if (iter == 0 && callback) {
            char opt_msg[256];
            snprintf(opt_msg, sizeof(opt_msg), "OPTIMIZER: Using %s optimizer (learning_rate=%.6f)",
                    model->adam_beta1 > 0.0f ? "Adam" : "Simple SGD (Python reference)", model->learning_rate);
            if (callback) callback("Optimizer Setup", iter, total_iters, 0.0f, opt_msg);
        }

        if (model->adam_beta1 > 0.0f) {
            // Adam optimizer (current default) - using double precision for numerical stability
            double adam_lr = static_cast<double>(model->learning_rate) *
                           std::sqrt(1.0 - std::pow(static_cast<double>(model->adam_beta2), iter + 1)) /
                           (1.0 - std::pow(static_cast<double>(model->adam_beta1), iter + 1));

            // ERROR14 Step 3: Check if SIMD is available via runtime detection
            bool use_simd = pacmap_simd::should_use_simd() && model->n_components >= 4;

            if (use_simd) {
                // FIX17.md Step 5: SIMD-optimized Adam optimizer using Eigen with better chunk size
                // Process samples in parallel, but vectorize within each sample's dimensions
                #pragma omp parallel for schedule(static, 1000)
                for (int sample = 0; sample < model->n_samples; ++sample) {
                    size_t sample_offset = static_cast<size_t>(sample) * model->n_components;

                    // Create Eigen maps for vectorized operations
                    Eigen::Map<Eigen::VectorXd> emb_vec(embedding.data() + sample_offset, model->n_components);
                    Eigen::Map<Eigen::VectorXd> grad_vec(gradients.data() + sample_offset, model->n_components);
                    Eigen::Map<Eigen::VectorXd> m_vec(model->adam_m.data() + sample_offset, model->n_components);
                    Eigen::Map<Eigen::VectorXd> v_vec(model->adam_v.data() + sample_offset, model->n_components);

                    // ERROR14 Step 3: Vectorized Adam state updates
                    // First check for NaN/Inf in gradients (deterministic order)
                    bool has_finite_gradients = true;
                    for (int d = 0; d < model->n_components; ++d) {
                        if (!std::isfinite(grad_vec(d))) {
                            has_finite_gradients = false;
                            break;
                        }
                    }

                    if (has_finite_gradients) {
                        // Update biased first moment estimate (vectorized)
                        m_vec = static_cast<double>(model->adam_beta1) * m_vec +
                                (1.0 - static_cast<double>(model->adam_beta1)) * grad_vec;

                        // Update biased second moment estimate (vectorized)
                        v_vec = static_cast<double>(model->adam_beta2) * v_vec +
                                (1.0 - static_cast<double>(model->adam_beta2)) * grad_vec.array().square().matrix();

                        // Ensure adam_v stays non-negative (vectorized)
                        v_vec = v_vec.cwiseMax(0.0);

                        // Compute Adam update with bias correction (vectorized)
                        Eigen::VectorXd adam_update_vec = adam_lr * m_vec.array() /
                                                        (v_vec.array().sqrt() + static_cast<double>(model->adam_eps));

                        // Check if Adam update is finite (deterministic order)
                        bool has_finite_updates = true;
                        for (int d = 0; d < model->n_components; ++d) {
                            if (!std::isfinite(adam_update_vec(d))) {
                                has_finite_updates = false;
                                break;
                            }
                        }

                        if (has_finite_updates) {
                            // Update parameters (vectorized)
                            emb_vec -= adam_update_vec;
                        }

                        // Final safety - reset non-finite embedding coordinates (deterministic)
                        for (int d = 0; d < model->n_components; ++d) {
                            if (!std::isfinite(emb_vec(d))) {
                                emb_vec(d) = 0.01 * ((d % 2 == 0) ? 1.0 : -1.0);
                            }
                        }
                    }
                }
            } else {
                // FIX17.md Step 5: Scalar fallback for non-AVX2 CPUs or small dimensions
                // Use schedule(static, 1000) for better cache locality
                #pragma omp parallel for schedule(static, 1000)
                for (int i = 0; i < static_cast<int>(embedding.size()); ++i) {
                    // ERROR13: NaN safety - skip non-finite gradients to prevent Adam state corruption
                    if (!std::isfinite(gradients[i])) {
                        continue;
                    }

                    // Update biased first moment estimate with RAW gradients (pure double precision)
                    model->adam_m[i] = static_cast<double>(model->adam_beta1) * model->adam_m[i] +
                                            (1.0 - static_cast<double>(model->adam_beta1)) * gradients[i];

                    // Update biased second moment estimate with RAW gradients (pure double precision)
                    model->adam_v[i] = static_cast<double>(model->adam_beta2) * model->adam_v[i] +
                                            (1.0 - static_cast<double>(model->adam_beta2)) * gradients[i] * gradients[i];

                    // ERROR13: Ensure adam_v stays non-negative (numerical safety check)
                    if (model->adam_v[i] < 0.0) {
                        model->adam_v[i] = 0.0;
                    }

                    // Compute Adam update with bias correction (pure double precision)
                    double adam_update = adam_lr * model->adam_m[i] /
                                        (std::sqrt(model->adam_v[i]) + static_cast<double>(model->adam_eps));

                    // ERROR13: Check if Adam update is finite before applying (prevents NaN spreading)
                    if (!std::isfinite(adam_update)) {
                        continue;
                    }

                    // Update parameters (pure double precision)
                    embedding[i] -= adam_update;

                    // ERROR13: Final safety - reset embedding coordinate if it becomes non-finite
                    if (!std::isfinite(embedding[i])) {
                        // Reset to small random value if it becomes non-finite
                        embedding[i] = 0.01 * ((i % 2 == 0) ? 1.0 : -1.0);
                    }
                }
            }
        } else {
            // Simple SGD optimizer (exact Python reference match)
            // Python line 351-352: embedding -= learning_rate * gradients

            // ERROR14 Step 3: Check if SIMD is available via runtime detection
            bool use_simd = pacmap_simd::should_use_simd() && model->n_components >= 4;

            if (use_simd) {
                // FIX17.md Step 5: SIMD-optimized SGD optimizer using Eigen with better chunk size
                // Process samples in parallel, but vectorize within each sample's dimensions
                #pragma omp parallel for schedule(static, 1000)
                for (int sample = 0; sample < model->n_samples; ++sample) {
                    size_t sample_offset = static_cast<size_t>(sample) * model->n_components;

                    // Create Eigen maps for vectorized operations
                    Eigen::Map<Eigen::VectorXd> emb_vec(embedding.data() + sample_offset, model->n_components);
                    Eigen::Map<Eigen::VectorXd> grad_vec(gradients.data() + sample_offset, model->n_components);

                    // ERROR14 Step 3: Vectorized SGD update
                    // First check for NaN/Inf in gradients (deterministic order)
                    bool has_finite_gradients = true;
                    for (int d = 0; d < model->n_components; ++d) {
                        if (!std::isfinite(grad_vec(d))) {
                            has_finite_gradients = false;
                            break;
                        }
                    }

                    if (has_finite_gradients) {
                        // Simple SGD update exactly like Python: embedding -= learning_rate * gradients (vectorized)
                        emb_vec -= static_cast<double>(model->learning_rate) * grad_vec;

                        // Safety check for non-finite embedding values (deterministic)
                        for (int d = 0; d < model->n_components; ++d) {
                            if (!std::isfinite(emb_vec(d))) {
                                emb_vec(d) = 0.01 * ((d % 2 == 0) ? 1.0 : -1.0);
                            }
                        }
                    }
                }
            } else {
                // FIX17.md Step 5: Scalar fallback for non-AVX2 CPUs or small dimensions
                // Use schedule(static, 1000) for better cache locality
                #pragma omp parallel for schedule(static, 1000)
                for (int i = 0; i < static_cast<int>(embedding.size()); ++i) {
                    // NaN safety check
                    if (!std::isfinite(gradients[i])) {
                        continue;
                    }

                    // Simple SGD update exactly like Python: embedding -= learning_rate * gradients
                    embedding[i] -= static_cast<double>(model->learning_rate) * gradients[i];

                    // Safety check for non-finite embedding values
                    if (!std::isfinite(embedding[i])) {
                        embedding[i] = 0.01 * ((i % 2 == 0) ? 1.0 : -1.0);
                    }
                }
            }
        }

        // Enhanced progress monitoring for large datasets - report every epoch for >100k samples
        bool should_report = (model->n_samples > 100000) || (iter % 10 == 0 || iter == total_iters - 1);

        if (should_report) {
            // MEMORY FIX: Use flat storage loss computation to prevent allocation failures
            double loss = compute_pacmap_loss_flat(embedding, model->triplets_flat,
                                                  w_n, w_mn, w_f, model->n_components);
            loss_history.push_back(static_cast<float>(loss));

            // DEBUG: Check embedding spread for large datasets (cross pattern detection)
            if (model->n_samples > 50000 && (iter == 0 || iter == 100 || iter == 500 || iter % 1000 == 0)) {
                double min_x = embedding[0], max_x = embedding[0], min_y = embedding[1], max_y = embedding[1];
                double sum_x = 0.0, sum_y = 0.0;

                for (int i = 0; i < model->n_samples; ++i) {
                    double x = embedding[i * 2];
                    double y = embedding[i * 2 + 1];

                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);

                    sum_x += x;
                    sum_y += y;
                }

                double center_x = sum_x / model->n_samples;
                double center_y = sum_y / model->n_samples;
                double spread_x = max_x - min_x;
                double spread_y = max_y - min_y;

                // Check for cross pattern: points clustered in cross formation
                double cross_score = 0.0;
                int points_near_axes = 0;
                for (int i = 0; i < model->n_samples; ++i) {
                    double x = embedding[i * 2];
                    double y = embedding[i * 2 + 1];

                    // Distance from center
                    double dx = x - center_x;
                    double dy = y - center_y;

                    // Check if point is near either axis (characteristic of cross pattern)
                    if (std::abs(dx) < 0.1 * spread_x || std::abs(dy) < 0.1 * spread_y) {
                        points_near_axes++;
                    }
                }

                double axis_ratio = static_cast<double>(points_near_axes) / model->n_samples;

              }


            monitor_optimization_progress(iter, total_iters, loss, get_current_phase(iter, model->phase1_iters, model->phase2_iters), callback);

            // Early termination check
            if (iter > 200 && should_terminate_early(loss_history)) {
                if (callback) callback("Early Termination - Converged", iter, total_iters,
                        static_cast<float>(iter) / total_iters * 100.0f,
                        "Convergence detected");
                break;
            }
        }
    }

    auto loop_end = std::chrono::high_resolution_clock::now();
    auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);
    if (!loss_history.empty()) {
    }

    // Compute final safety statistics
    compute_safety_stats(model, embedding);

    // Save embedding in model for transform operations (convert to float for storage compatibility)
    model->embedding.resize(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        model->embedding[i] = static_cast<float>(embedding[i]);
    }

    // Copy results to output (keep as double precision)
    for (size_t i = 0; i < embedding.size(); ++i) {
        embedding_out[i] = embedding[i];
    }

    if (callback) callback("Optimization Complete", total_iters, total_iters, 100.0f, nullptr);
}

void initialize_random_embedding_double(std::vector<double>& embedding, int n_samples, int n_components, pcg64_fast& rng, float std_dev) {
    // pcg64_fast-based random initialization
    std::normal_distribution<double> normal_dist(0.0, static_cast<double>(std_dev));

    for (int i = 0; i < n_samples; ++i) {
        for (int d = 0; d < n_components; ++d) {
            size_t idx = static_cast<size_t>(i) * n_components + d;
            embedding[idx] = normal_dist(rng);
        }
    }
}


void initialize_random_embedding(std::vector<float>& embedding, int n_samples, int n_components, pcg64_fast& rng, float std_dev) {
    // pcg64_fast-based random initialization using double precision internally
    std::vector<double> embedding_double(n_samples * n_components);
    initialize_random_embedding_double(embedding_double, n_samples, n_components, rng, std_dev);

    // Convert back to float32 for compatibility
    for (size_t i = 0; i < embedding_double.size(); ++i) {
        embedding[i] = static_cast<float>(embedding_double[i]);
    }
}


// REMOVED: run_optimization_phase function - unused dead code that took unused Adam state parameters
// REMOVED: Old AdaGrad optimization functions - now using Adam optimizer in main optimize_embedding function

std::vector<OptimizationPhase> get_optimization_phases(int phase1_iters, int phase2_iters, int phase3_iters) {
    std::vector<OptimizationPhase> phases;

    // Phase 1: Global structure focus
    phases.push_back({"Phase 1: Global Structure", 0, phase1_iters, 3.0f, 1.0f, 1.0f});

    // Phase 2: Balance phase
    phases.push_back({"Phase 2: Balance", phase1_iters, phase1_iters + phase2_iters, 3.0f, 1.0f, 1.0f});

    // Phase 3: Local structure focus
    phases.push_back({"Phase 3: Local Structure", phase1_iters + phase2_iters,
                     phase1_iters + phase2_iters + phase3_iters, 0.0f, 1.0f, 1.0f});

    return phases;
}

void compute_safety_stats(PacMapModel* model, const std::vector<double>& embedding) {
    // Compute pairwise embedding distances for safety analysis (double precision)
    std::vector<float> embedding_distances;

    // Sample distances for efficiency (similar to UMAP approach)
    int sample_size = static_cast<int>(std::min(model->n_samples, static_cast<int64_t>(1000)));
    for (int i = 0; i < sample_size; ++i) {
        for (int j = i + 1; j < sample_size; ++j) {
            // Compute distance in double precision, store as float for stats
            double dist_double = distance_metrics::compute_distance(
                embedding.data() + i * model->n_components,
                embedding.data() + j * model->n_components,
                model->n_components, PACMAP_METRIC_EUCLIDEAN);
            embedding_distances.push_back(static_cast<float>(dist_double));
        }
    }

    // Compute percentiles for outlier detection
    std::sort(embedding_distances.begin(), embedding_distances.end());
    model->min_embedding_dist = embedding_distances[0];
    model->p95_embedding_dist = embedding_distances[embedding_distances.size() * 0.95];
}

void monitor_optimization_progress(int iter, int total_iters, float loss,
                                  const std::string& phase, pacmap_progress_callback_internal callback) {
    // Map optimization progress to 50-90% range (steps 5-10 of overall process)
    float base_progress = 50.0f;  // Start at 50% after setup
    float optimization_range = 40.0f;  // 40% of total progress for optimization
    float progress = base_progress + (static_cast<float>(iter) / total_iters) * optimization_range;

    // Format loss with appropriate precision
    std::ostringstream loss_stream;
    loss_stream << std::fixed << std::setprecision(6) << loss;

    // Create detailed progress message
    std::string message = phase + " (Iter " + std::to_string(iter) + "/" + std::to_string(total_iters) +
                           ") - Loss: " + loss_stream.str();

    if (callback) callback(phase.c_str(), iter, total_iters, progress, message.c_str());
}

float compute_embedding_quality(const std::vector<float>& embedding, const std::vector<uint64_t>& triplets_flat,
                               int n_components) {
    if (triplets_flat.empty()) return 0.0f;

    float total_quality = 0.0f;
    size_t num_triplets = triplets_flat.size() / 3;

    for (size_t idx = 0; idx < num_triplets; ++idx) {
        size_t triplet_offset = idx * 3;

        // Extract triplet data from flat storage
        uint64_t anchor = triplets_flat[triplet_offset];
        uint64_t neighbor = triplets_flat[triplet_offset + 1];
        uint64_t type = triplets_flat[triplet_offset + 2];

        size_t idx_a = static_cast<size_t>(anchor) * n_components;
        size_t idx_n = static_cast<size_t>(neighbor) * n_components;

        float dist = compute_sampling_distance(embedding.data() + idx_a,
                                             embedding.data() + idx_n,
                                             n_components, PACMAP_METRIC_EUCLIDEAN);

        // Quality metric based on triplet type and distance
        float quality;
        switch (static_cast<TripletType>(type)) {
            case NEIGHBOR:
                quality = 1.0f / (1.0f + dist);  // Should be small
                break;
            case MID_NEAR:
                quality = dist / (100.0f + dist);  // Should be medium
                break;
            case FURTHER:
                quality = dist / (1.0f + dist);   // Should be large
                break;
        }

        total_quality += quality;
    }

    return total_quality / static_cast<float>(num_triplets);
}

bool check_optimization_convergence(const std::vector<float>& loss_history, float tolerance) {
    return check_convergence(loss_history, tolerance, 50);
}

void apply_optimization_strategy(PacMapModel* model, OptimizationStrategy strategy) {
    switch (strategy) {
        case OptimizationStrategy::FAST:
            model->learning_rate *= 1.5f;
            model->phase1_iters = static_cast<int>(model->phase1_iters * 0.7f);
            model->phase2_iters = static_cast<int>(model->phase2_iters * 0.7f);
            model->phase3_iters = static_cast<int>(model->phase3_iters * 0.7f);
            break;

        case OptimizationStrategy::ACCURATE:
            model->learning_rate *= 0.8f;
            model->phase1_iters = static_cast<int>(model->phase1_iters * 1.3f);
            model->phase2_iters = static_cast<int>(model->phase2_iters * 1.3f);
            model->phase3_iters = static_cast<int>(model->phase3_iters * 1.3f);
            break;

        case OptimizationStrategy::MEMORY_EFFICIENT:
            // Reduce triplet counts for memory efficiency
            // ERROR11-FIX-DETERMINISTIC: Do not modify mn_ratio - it should come from C# interface
            // model->mn_ratio *= 0.8f;  // REMOVED: Prevents C# parameter control
            model->fp_ratio *= 0.8f;
            break;

        case OptimizationStrategy::STANDARD:
        default:
            // Use default parameters
            break;
    }
}

void adaptive_phase_transitions(PacMapModel* model, std::vector<float>& loss_history) {
    if (loss_history.size() < 20) return;

    // Check if current phase is effective
    float recent_loss_change = loss_history.back() - loss_history[loss_history.size() - 20];

    if (std::abs(recent_loss_change) < 1e-8f) {
        // Loss not changing significantly, consider phase transition
        // Implementation would adjust phase boundaries dynamically
    }
}

void dynamic_learning_rate_adjustment(PacMapModel* model, int iter, float current_loss) {
    // Adjust learning rate based on loss behavior
    static float prev_loss = current_loss;
    static int stagnation_count = 0;

    if (std::abs(current_loss - prev_loss) < 1e-8f) {
        stagnation_count++;
        if (stagnation_count > 10) {
            model->learning_rate *= 0.9f;  // Reduce learning rate
            stagnation_count = 0;
        }
    } else {
        stagnation_count = 0;
        // Can slightly increase learning rate if making good progress
        if (current_loss < prev_loss) {
            model->learning_rate *= 1.02f;
        }
    }

    prev_loss = current_loss;
}

void early_stopping_check(PacMapModel* model, const std::vector<float>& loss_history, int patience) {
    if (should_terminate_early(loss_history, patience)) {
        // Early stopping would be triggered
    }
}

OptimizationDiagnostics run_optimization_with_diagnostics(PacMapModel* model,
                                                       std::vector<float>& embedding,
                                                       pacmap_progress_callback_internal callback) {
    OptimizationDiagnostics diagnostics;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert float embedding to double for computation
    std::vector<double> embedding_double(embedding.size());
    for (size_t i = 0; i < embedding.size(); i++) {
        embedding_double[i] = static_cast<double>(embedding[i]);
    }

    // Store initial loss
    // FIX21: Use iteration-based weight function
    int total_iters = model->phase1_iters + model->phase2_iters + model->phase3_iters;
    auto [w_n, w_mn, w_f] = get_weights(0, model->phase1_iters, model->phase2_iters);
    // MEMORY FIX: Use flat storage loss computation for diagnostics
    diagnostics.initial_loss = static_cast<float>(compute_pacmap_loss_flat(embedding_double, model->triplets_flat,
                                                  w_n, w_mn, w_f, model->n_components));

    // Run optimization with detailed monitoring (using double precision)
    optimize_embedding(model, embedding_double.data(), callback);

    // Convert result back to float for output
    for (size_t i = 0; i < embedding.size(); i++) {
        embedding[i] = static_cast<float>(embedding_double[i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    diagnostics.optimization_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Store final loss and compute reduction
    // MEMORY FIX: Use flat storage loss computation for diagnostics
    diagnostics.final_loss = static_cast<float>(compute_pacmap_loss_flat(embedding_double, model->triplets_flat,
                                                w_n, w_mn, w_f, model->n_components));
    diagnostics.loss_reduction = diagnostics.initial_loss - diagnostics.final_loss;

    return diagnostics;
}

std::string get_current_phase(int iter, int phase1_iters, int phase2_iters) {
    if (iter < phase1_iters) {
        return "Phase 1: Global Structure";
    } else if (iter < phase1_iters + phase2_iters) {
        return "Phase 2: Balance";
    } else {
        return "Phase 3: Local Structure";
    }
}

float get_phase_weight(int iter, int total_iters, float start_weight, float end_weight) {
    float progress = static_cast<float>(iter) / total_iters;
    return start_weight + (end_weight - start_weight) * progress;
}

void assess_embedding_quality(const std::vector<float>& embedding, const PacMapModel* model) {
    float quality = compute_embedding_quality(embedding, model->triplets_flat, model->n_components);
}

// REMOVED: compute_trustworthiness function - would need flat storage adaptation if needed

void optimize_for_dataset_size(PacMapModel* model, int n_samples) {
    if (n_samples > 10000) {
        // Large dataset optimization
        model->use_quantization = true;
        model->hnsw_ef_construction = 100;  // Faster index building
    } else if (n_samples < 1000) {
        // Small dataset optimization
        model->learning_rate *= 0.8f;  // More conservative learning
    }
}

void configure_memory_usage(PacMapModel* model, size_t available_memory_mb) {
    size_t estimated_memory_mb = model->n_samples * model->n_features * 4 / (1024 * 1024);  // Rough estimate

    if (estimated_memory_mb > available_memory_mb * 0.8f) {
        model->use_quantization = true;
            }
}

void validate_optimization_state(const PacMapModel* model, const std::vector<float>& embedding) {
    if (model->triplets_flat.empty()) {
        std::cerr << "Warning: No triplets available for optimization" << std::endl;
    }

    if (embedding.size() != static_cast<size_t>(model->n_samples * model->n_components)) {
        std::cerr << "Error: Embedding size mismatch" << std::endl;
    }

    // Check for NaN/Inf in embedding
    int nan_count = 0, inf_count = 0;
    for (float val : embedding) {
        if (std::isnan(val)) nan_count++;
        if (std::isinf(val)) inf_count++;
    }

    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "Warning: " << nan_count << " NaN and " << inf_count << " Inf values in embedding" << std::endl;
    }
}

void print_optimization_summary(const OptimizationDiagnostics& diagnostics) {
    // Summary would be reported via callback in production
}