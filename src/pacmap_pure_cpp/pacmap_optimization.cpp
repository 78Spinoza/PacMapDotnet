#include "pacmap_optimization.h"
#include "pacmap_gradient.h"
#include "pacmap_triplet_sampling.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <chrono>

void optimize_embedding(PacMapModel* model, float* embedding_out, pacmap_progress_callback_internal callback) {

    std::vector<float> embedding(model->n_samples * model->n_components);

    // Initialize embedding with proper random normal distribution using model's initialization_std_dev parameter
    initialize_random_embedding(embedding, model->n_samples, model->n_components, model->rng, model->initialization_std_dev);

    int total_iters = model->phase1_iters + model->phase2_iters + model->phase3_iters;

    if (model->triplets.empty()) {
        if (callback) callback("Error", 0, 100, 0.0f, "No triplets available for optimization");
        return;
    }

    // Initialize Adam optimizer state
    std::vector<float> gradients(embedding.size());

    // Loss history for convergence monitoring
    std::vector<float> loss_history;
    loss_history.reserve(total_iters);

    // Initialize Adam optimizer state in model
    model->adam_m.assign(embedding.size(), 0.0f);
    model->adam_v.assign(embedding.size(), 0.0f);

                
    callback("Starting Optimization", 0, 0, 0.0f, nullptr);

    auto loop_start = std::chrono::high_resolution_clock::now();

    // Main optimization loop with three phases
    for (int iter = 0; iter < total_iters; ++iter) {
        // Get three-phase weights for current iteration
        auto [w_n, w_mn, w_f] = get_weights(iter, model->phase1_iters, model->phase1_iters + model->phase2_iters);

        // Compute gradients for all triplets
        compute_gradients(embedding, model->triplets, gradients,
                         w_n, w_mn, w_f, model->n_components);

        // CRITICAL FIX: Adam optimizer with proper gradient handling (FIXED GRADIENT CLIPPING ORDER)
        // Adam optimizer with bias correction (matching Rust implementation)
        float adam_lr = model->learning_rate *
                       std::sqrt(1.0f - std::pow(model->adam_beta2, iter + 1)) /
                       (1.0f - std::pow(model->adam_beta1, iter + 1));

        
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(embedding.size()); ++i) {
            // Update biased first moment estimate with RAW gradients (critical!)
            model->adam_m[i] = model->adam_beta1 * model->adam_m[i] +
                              (1.0f - model->adam_beta1) * gradients[i];

            // Update biased second moment estimate with RAW gradients (critical!)
            model->adam_v[i] = model->adam_beta2 * model->adam_v[i] +
                              (1.0f - model->adam_beta2) * gradients[i] * gradients[i];

            // Compute Adam update with bias correction
            float adam_update = adam_lr * model->adam_m[i] / (std::sqrt(model->adam_v[i]) + model->adam_eps);

            // CRITICAL FIX: Removed gradient clipping to match Rust implementation exactly
            // Rust PACMAP does not apply gradient clipping - let Adam optimizer handle scale naturally

            // Update parameters
            embedding[i] -= adam_update;
        }

        // Monitor progress and compute loss
        if (iter % 10 == 0 || iter == total_iters - 1) {
            float loss = compute_pacmap_loss(embedding, model->triplets,
                                           w_n, w_mn, w_f, model->n_components);
            loss_history.push_back(loss);

            std::string phase = get_current_phase(iter, model->phase1_iters, model->phase2_iters);
            monitor_optimization_progress(iter, total_iters, loss, phase, callback);

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

    // Save embedding in model for transform operations
    model->embedding = embedding;

    // Copy results to output
    std::memcpy(embedding_out, embedding.data(), embedding.size() * sizeof(float));

    callback("Optimization Complete", total_iters, total_iters, 100.0f, nullptr);
}

void initialize_random_embedding(std::vector<float>& embedding, int n_samples, int n_components, std::mt19937& rng, float std_dev) {
    // CRITICAL FIX: Use provided initialization_std_dev parameter instead of hardcoded large value
    // The previous value of 10.0f / sqrt(n_components) ≈ 5.77 was too large and caused fragmentation
    // Now using the parameter provided through the API (default 0.1f) for proper initialization
    std::normal_distribution<float> normal_dist(0.0f, std_dev);

    for (auto& val : embedding) {
        val = normal_dist(rng);
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

void compute_safety_stats(PacMapModel* model, const std::vector<float>& embedding) {
    // Compute pairwise embedding distances for safety analysis
    std::vector<float> embedding_distances;

    // Sample distances for efficiency (similar to UMAP approach)
    int sample_size = std::min(model->n_samples, 1000);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = i + 1; j < sample_size; ++j) {
            // FIXED: Use Euclidean metric consistently (error analysis #4)
            float dist = compute_sampling_distance(embedding.data() + i * model->n_components,
                                                 embedding.data() + j * model->n_components,
                                                 model->n_components, PACMAP_METRIC_EUCLIDEAN);
            embedding_distances.push_back(dist);
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

    // Store initial loss
    auto [w_n, w_mn, w_f] = get_weights(0, model->phase1_iters, model->phase1_iters + model->phase2_iters);
    diagnostics.initial_loss = compute_pacmap_loss(embedding, model->triplets,
                                                  w_n, w_mn, w_f, model->n_components);

    // Run optimization with detailed monitoring
    optimize_embedding(model, embedding.data(), callback);

    auto end_time = std::chrono::high_resolution_clock::now();
    diagnostics.optimization_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Store final loss and compute reduction
    diagnostics.final_loss = compute_pacmap_loss(embedding, model->triplets,
                                                w_n, w_mn, w_f, model->n_components);
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
        std::cout << "Enabling quantization due to memory constraints" << std::endl;
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