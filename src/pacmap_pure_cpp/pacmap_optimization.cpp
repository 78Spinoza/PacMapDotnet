#include "pacmap_optimization.h"
#include "pacmap_gradient.h"
#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <chrono>

void optimize_embedding(PacMapModel* model, double* embedding_out, pacmap_progress_callback_internal callback) {

    // Python hybrid approach: Use double precision internally for numerical stability
    std::vector<double> embedding(model->n_samples * model->n_components);

    // Initialize embedding with proper random normal distribution using model's initialization_std_dev parameter
    // Python reference: np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    if (callback) {
        char init_msg[256];
        snprintf(init_msg, sizeof(init_msg), "INITIALIZATION: Using std_dev=%.6f for %d samples x %d components",
                model->initialization_std_dev, model->n_samples, model->n_components);
        callback("Initialization", 0, 0, 0.0f, init_msg);
    }
    initialize_random_embedding_double(embedding, model->n_samples, model->n_components, model->rng, model->initialization_std_dev);

    
    int total_iters = model->phase1_iters + model->phase2_iters + model->phase3_iters;

    if (model->triplets.empty()) {
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

                
    callback("Starting Optimization", 0, 0, 0.0f, nullptr);

    auto loop_start = std::chrono::high_resolution_clock::now();

    // Main optimization loop with three phases
    for (int iter = 0; iter < total_iters; ++iter) {
        // ERROR13 FIX: Get three-phase weights for current iteration using Python-matching progress
        auto [w_n, w_mn, w_f] = get_weights(iter, total_iters);

        // Compute gradients for all triplets - now using double precision embedding with callback
        compute_gradients(embedding, model->triplets, gradients,
                         w_n, w_mn, w_f, model->n_components, callback);

        // ERROR13 FIX: Choose optimizer based on Python reference vs Adam optimization
        // Python reference uses simple SGD: embedding -= learning_rate * gradients
        // Our implementation defaults to Adam but allows SGD for exact Python matching

        if (iter == 0 && callback) {
            char opt_msg[256];
            snprintf(opt_msg, sizeof(opt_msg), "OPTIMIZER: Using %s optimizer (learning_rate=%.6f)",
                    model->adam_beta1 > 0.0f ? "Adam" : "Simple SGD (Python reference)", model->learning_rate);
            callback("Optimizer Setup", iter, total_iters, 0.0f, opt_msg);
        }

        if (model->adam_beta1 > 0.0f) {
            // Adam optimizer (current default) - using double precision for numerical stability
            double adam_lr = static_cast<double>(model->learning_rate) *
                           std::sqrt(1.0 - std::pow(static_cast<double>(model->adam_beta2), iter + 1)) /
                           (1.0 - std::pow(static_cast<double>(model->adam_beta1), iter + 1));

            #pragma omp parallel for
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
        } else {
            // Simple SGD optimizer (exact Python reference match)
            // Python line 351-352: embedding -= learning_rate * gradients
            #pragma omp parallel for
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

        // Monitor progress and compute loss
        if (iter % 10 == 0 || iter == total_iters - 1) {
            double loss = compute_pacmap_loss(embedding, model->triplets,
                                                   w_n, w_mn, w_f, model->n_components);
            loss_history.push_back(static_cast<float>(loss));

            
            monitor_optimization_progress(iter, total_iters, loss, get_current_phase(iter, model->phase1_iters, model->phase2_iters), callback);

            // Early termination check
            if (iter > 200 && should_terminate_early(loss_history)) {
                callback("Early Termination - Converged", iter, total_iters,
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

    callback("Optimization Complete", total_iters, total_iters, 100.0f, nullptr);
}

void initialize_random_embedding_double(std::vector<double>& embedding, int n_samples, int n_components, std::mt19937& rng, float std_dev) {
    // Legacy function - kept for compatibility but now uses NumPy RNG through model
    // This function should not be used - prefer the version that takes NumpyRandom
    std::normal_distribution<double> normal_dist(0.0, static_cast<double>(std_dev));

    for (int i = 0; i < n_samples; ++i) {
        for (int d = 0; d < n_components; ++d) {
            size_t idx = static_cast<size_t>(i) * n_components + d;
            embedding[idx] = normal_dist(rng);
        }
    }
}


void initialize_random_embedding(std::vector<float>& embedding, int n_samples, int n_components, std::mt19937& rng, float std_dev) {
    // Legacy function for compatibility - now forwards to double precision version
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
    int sample_size = std::min(model->n_samples, 1000);
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

    callback(phase.c_str(), iter, total_iters, progress, message.c_str());
}

float compute_embedding_quality(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                               int n_components) {
    if (triplets.empty()) return 0.0f;

    float total_quality = 0.0f;

    for (const auto& triplet : triplets) {
        size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

        float dist = compute_sampling_distance(embedding.data() + idx_a,
                                             embedding.data() + idx_n,
                                             n_components, PACMAP_METRIC_EUCLIDEAN);

        // Quality metric based on triplet type and distance
        float quality;
        switch (triplet.type) {
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

    return total_quality / static_cast<float>(triplets.size());
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
    // ERROR13 FIX: Use corrected weight function signature
    int total_iters = model->phase1_iters + model->phase2_iters + model->phase3_iters;
    auto [w_n, w_mn, w_f] = get_weights(0, total_iters);
    diagnostics.initial_loss = static_cast<float>(compute_pacmap_loss(embedding_double, model->triplets,
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
    diagnostics.final_loss = static_cast<float>(compute_pacmap_loss(embedding_double, model->triplets,
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
    float quality = compute_embedding_quality(embedding, model->triplets, model->n_components);
}

float compute_trustworthiness(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                             int n_samples, int n_components) {
    // Simplified trustworthiness metric
    // In practice, this would compare k-nearest neighbors in original vs embedded space
    int preserved_triplets = 0;

    for (const auto& triplet : triplets) {
        if (triplet.type == NEIGHBOR) {
            // Check if neighbor pairs are still close in embedding space
            size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
            size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

            float emb_dist = compute_sampling_distance(embedding.data() + idx_a,
                                                     embedding.data() + idx_n,
                                                     n_components, PACMAP_METRIC_EUCLIDEAN);

            if (emb_dist < 0.1f) {  // Arbitrary threshold
                preserved_triplets++;
            }
        }
    }

    int neighbor_triplets = std::count_if(triplets.begin(), triplets.end(),
                                         [](const Triplet& t) { return t.type == NEIGHBOR; });

    return neighbor_triplets > 0 ? static_cast<float>(preserved_triplets) / neighbor_triplets : 0.0f;
}

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
    if (model->triplets.empty()) {
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