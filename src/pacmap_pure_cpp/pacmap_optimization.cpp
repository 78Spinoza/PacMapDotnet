#include "pacmap_optimization.h"
#include "pacmap_gradient.h"
#include "pacmap_triplet_sampling.h"
#include <random>
#include <iostream>
#include <iomanip>
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

    // CRITICAL DEBUG: Print exact hyperparameters and KNN mode
    float std_dev = model->initialization_std_dev;
    printf("[OPTIM DEBUG] Starting: total_iters=%d, init_std_dev=%.6f, n_triplets=%zu\n",
           total_iters, std_dev, model->triplets.size());
    printf("\n=== PACMAP OPTIMIZATION START (ADAM v2.0.0 - GRADIENT-FIXED) ===\n");
    printf("🔥 CONFIRMED: Using ADAM OPTIMIZER with bias correction!\n");
    printf("🔥 CRITICAL FIXES APPLIED: No gradient normalization, w_MN=0 in Phase 3, std_dev=1e-4\n");
    printf("Hyperparameters:\n");
    printf("  Learning Rate: %.6f\n", model->learning_rate);
    printf("  Adam Beta1: %.3f, Beta2: %.3f, Epsilon: %.2e\n",
           model->adam_beta1, model->adam_beta2, model->adam_eps);
    printf("  N Neighbors: %d\n", model->n_neighbors);
    printf("  MN Ratio: %.2f\n", model->mn_ratio);
    printf("  FP Ratio: %.2f\n", model->fp_ratio);
    printf("  Distance Metric: %s\n", model->metric == PACMAP_METRIC_EUCLIDEAN ? "Euclidean" : "Cosine");
    printf("  Random Seed: %d\n", model->random_seed);
    printf("  KNN Mode: %s\n", model->force_exact_knn ? "Direct (Brute-Force)" : "HNSW");
    if (!model->force_exact_knn) {
        printf("  HNSW Parameters: M=%d, ef_construction=%d, ef_search=%d\n",
               model->hnsw_m, model->hnsw_ef_construction, model->hnsw_ef_search);
    }
    printf("  Phase Iterations: %d, %d, %d (Total: %d)\n",
           model->phase1_iters, model->phase2_iters, model->phase3_iters, total_iters);
    printf("  Data Points: %d, Dimensions: %d -> %d\n",
           model->n_samples, model->n_features, model->n_components);
    printf("  Initialization Std Dev: %.6f (model parameter)\n", std_dev);
    printf("  Triplets Generated: %zu\n", model->triplets.size());
    printf("=====================================\n\n");

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

        // ENHANCED DEBUG: Track optimization progress every 10 iterations
        if (iter % 10 == 0) {
            printf("[OPTIM DEBUG] Iter %d: lr=%.4e, w_n=%.1f, w_mn=%.1f, w_f=%.1f\n",
                   iter, adam_lr, w_n, w_mn, w_f);

            // Sample a few gradient statistics
            float grad_min = gradients[0], grad_max = gradients[0], grad_sum = 0.0f;
            int grad_nan = 0, grad_inf = 0;
            for (int i = 0; i < static_cast<int>(gradients.size()); ++i) {
                float g = gradients[i];
                grad_min = std::min(grad_min, g);
                grad_max = std::max(grad_max, g);
                grad_sum += g;
                if (std::isnan(g)) grad_nan++;
                if (std::isinf(g)) grad_inf++;
            }
            float grad_mean = grad_sum / gradients.size();

            // Only show gradient issues if they exist
            if (grad_nan > 0 || grad_inf > 0 || grad_max > 100.0f) {
                printf("   ⚠️  Gradient issues: min=%.6f, max=%.6f, nan=%d, inf=%d\n",
                       grad_min, grad_max, grad_nan, grad_inf);
            }

            // Sample Adam state statistics
            if (!model->adam_m.empty() && !model->adam_v.empty()) {
                float m_min = model->adam_m[0], m_max = model->adam_m[0], m_sum = 0.0f;
                float v_min = model->adam_v[0], v_max = model->adam_v[0], v_sum = 0.0f;
                int m_zero = 0, v_zero = 0, m_nan = 0, v_nan = 0;

                for (int i = 0; i < static_cast<int>(model->adam_m.size()); ++i) {
                    float m_val = model->adam_m[i];
                    float v_val = model->adam_v[i];

                    m_min = std::min(m_min, m_val);
                    m_max = std::max(m_max, m_val);
                    m_sum += m_val;
                    if (m_val == 0.0f) m_zero++;
                    if (std::isnan(m_val)) m_nan++;

                    v_min = std::min(v_min, v_val);
                    v_max = std::max(v_max, v_val);
                    v_sum += v_val;
                    if (v_val == 0.0f) v_zero++;
                    if (std::isnan(v_val)) v_nan++;
                }

                // Removed verbose Adam state debug for cleaner output
            }
        }

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

        // Monitor progress and compute loss - CRITICAL DEBUG: More frequent logging
        if (iter % 10 == 0 || iter == total_iters - 1) {
            float loss = compute_pacmap_loss(embedding, model->triplets,
                                           w_n, w_mn, w_f, model->n_components);
            loss_history.push_back(loss);

            // Enhanced debug for loss and gradient magnitude
            float delta_max = 0.0f;
            for (float g : gradients) {
                delta_max = std::max(delta_max, std::abs(g));
            }
            printf("[OPTIM DEBUG] Iter %d: loss=%.4e, delta_max=%.4e\n",
                   iter, loss, delta_max);

            std::string phase = get_current_phase(iter, model->phase1_iters, model->phase2_iters);

            // CRITICAL DEBUG: Check embedding spread
            float min_emb = embedding[0], max_emb = embedding[0];
            for (float val : embedding) {
                min_emb = std::min(min_emb, val);
                max_emb = std::max(max_emb, val);
            }

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
    float progress = static_cast<float>(iter) / total_iters * 100.0f;
    std::string message = phase + " - Loss: " + std::to_string(loss);
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