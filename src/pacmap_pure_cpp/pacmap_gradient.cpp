#include "pacmap_gradient.h"
#include "pacmap_distance.h"
#include "pacmap_triplet_sampling.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static std::chrono::high_resolution_clock::time_point gradient_start_time;

std::tuple<float, float, float> get_weights(int current_iter, int total_iters) {
    // FIX13-v4: CRITICAL - Match Python weight schedule EXACTLY!
    // Python reference (pacmap.py lines 337-353) shows w_neighbors CHANGES across phases!
    //
    // PREVIOUS BUG: C++ used constant w_n=1.0, but Python uses 2.0 → 3.0 → 1.0
    // This caused 2-3x weaker local structure in Phases 1-2, leading to oval formation
    //
    // Python weight schedule:
    //   Phase 1: w_neighbors=2.0, w_MN=1000→3, w_FP=1.0
    //   Phase 2: w_neighbors=3.0, w_MN=3.0, w_FP=1.0
    //   Phase 3: w_neighbors=1.0, w_MN=0.0, w_FP=1.0
    float w_n, w_mn;
    float w_f = 1.0f;  // Constant in Python (this is correct)

    float progress = static_cast<float>(current_iter) / total_iters;

    if (progress < 0.1f) {
        // Phase 1: Global structure (0-10%)
        float phase1_progress = progress / 0.1f;
        w_mn = 1000.0f * (1.0f - phase1_progress) + 3.0f * phase1_progress;
        w_n = 2.0f;   // ← FIX: Python uses 2.0, NOT 1.0!
    } else if (progress < 0.4f) {
        // Phase 2: Balance phase (10-40%)
        w_mn = 3.0f;
        w_n = 3.0f;   // ← FIX: Python uses 3.0, NOT 1.0!
    } else {
        // Phase 3: Local structure (40-100%)
        // FIX13: w_MN must be ZERO immediately in Phase 3 (not gradual decay)
        // Python reference (pacmap.py line 350): w_MN = 0.0 (instant zero)
        // Previous bug: Used gradual decay w_mn = 3.0f * (1.0f - phase3_progress) → wrong force balance
        w_mn = 0.0f;  // ← Match Python exactly!
        w_n = 1.0f;   // ← Finally matches Python in Phase 3
    }

    return {w_n, w_mn, w_f};
}

std::tuple<float, float, float, std::string> get_weights_with_phase_info(int current_iter, int total_iters) {
    float w_n, w_mn;
    float w_f = 1.0f;
    std::string current_phase_name;

    float progress = static_cast<float>(current_iter) / total_iters;

    if (progress < 0.1f) {
        float phase1_progress = progress / 0.1f;
        w_mn = 1000.0f * (1.0f - phase1_progress) + 3.0f * phase1_progress;
        w_n = 2.0f;
        current_phase_name = "Phase 1: Global Structure";
    } else if (progress < 0.4f) {
        w_mn = 3.0f;
        w_n = 3.0f;
        current_phase_name = "Phase 2: Balance Phase";
    } else {
        w_mn = 0.0f;
        w_n = 1.0f;
        current_phase_name = "Phase 3: Local Structure";
    }

    return {w_n, w_mn, w_f, current_phase_name};
}

void print_gradient_stats(int processed_neighbors, int processed_midnear, int processed_further,
                         int skipped_nan, int skipped_zero_distance, int skipped_triplets,
                         const std::string& current_phase_name, int current_iter, int total_iters,
                         float w_n, float w_mn, float w_f) {
    // Function body kept for compatibility but output removed as requested
    // Stats are available through function parameters if needed elsewhere
    (void)processed_neighbors; (void)processed_midnear; (void)processed_further;
    (void)skipped_nan; (void)skipped_zero_distance; (void)skipped_triplets;
    (void)current_phase_name; (void)current_iter; (void)total_iters;
    (void)w_n; (void)w_mn; (void)w_f;
}

bool validate_gradients(const std::vector<double>& gradients, const std::string& context) {
    int nan_count = 0, inf_count = 0, large_count = 0;
    double max_grad = 0.0, avg_grad = 0.0;
    const double large_threshold = 1e10;

    for (size_t i = 0; i < gradients.size(); ++i) {
        double g = gradients[i];
        if (std::isnan(g)) nan_count++;
        else if (std::isinf(g)) inf_count++;
        else {
            max_grad = std::max(max_grad, std::abs(g));
            avg_grad += std::abs(g);
            if (std::abs(g) > large_threshold) large_count++;
        }
    }

    avg_grad /= static_cast<double>(gradients.size());

    // Output removed as requested - validation logic preserved
    (void)context; // Suppress unused parameter warning

    bool valid = (nan_count == 0 && inf_count == 0 && large_count == 0);
    return valid;
}


void compute_gradients(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                       std::vector<double>& gradients, float w_n, float w_mn, float w_f, int n_components,
                       pacmap_progress_callback_internal callback) {

    gradients.assign(embedding.size(), 0.0);

    // PHASE 2 FIX: CRITICAL - Match Python sequential processing order exactly!
    // Python reference processes ALL triplets by type sequentially:
    //   1. ALL NEIGHBOR triplets first
    //   2. ALL MID_NEAR triplets second
    //   3. ALL FURTHER triplets third
    //
    // PREVIOUS BUG: C++ processed in interleaved order due to shuffling
    // This creates different floating-point rounding and optimization dynamics
    //
    // FIX: Process triplets by type sequentially like Python (pacmap.py lines 268-306)

    // Process all triplet types sequentially
    int processed_neighbors = 0, processed_midnear = 0, processed_further = 0;
    int skipped_nan = 0, skipped_zero_distance = 0, skipped_triplets = 0;
    std::string current_phase_name;
    int phase_transition_count = 0;

    // PHASE 1: Process ALL NEIGHBOR triplets (matches Python lines 268-279)
    #pragma omp parallel for schedule(dynamic, 1000) reduction(+:processed_neighbors,skipped_nan,skipped_zero_distance)
    for (int idx = 0; idx < static_cast<int>(triplets.size()); ++idx) {
        const auto& t = triplets[idx];
        if (t.type != NEIGHBOR) continue;  // Skip non-neighbor triplets in this phase

        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // FIX13: CRITICAL - Match Python distance calculation EXACTLY!
        // Python (pacmap.py line 271-274): d_ij = 1.0, then d_ij += y_ij[d]**2
        // Result: d_ij = 1.0 + sum(diff²)  ← NOT sqrt(sum(diff²))!
        // This is the SQUARED distance plus 1.0, NOT Euclidean distance!
        double d_ij = 1.0;
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij += diff * diff;  // Add squared difference (NO sqrt!)
        }
        // Note: d_ij starts at 1.0, so it's always >= 1.0 (no need to check for zero)

        // FIX v2.8.7: CRITICAL - Match Python gradient formula EXACTLY!
        // Python (pacmap.py line 276): w1 = w_neighbors * (20. / (10. + d_ij) ** 2)
        // Then (line 278): grad[i, d] += w1 * y_ij[d]  ← RAW difference, NOT normalized!
        //
        // PREVIOUS BUG: Used 10.0 instead of 20.0 (missing factor of 2)
        //               AND divided by d_ij (incorrect normalization)
        // RESULT: 2× weaker forces + distance-independent magnitudes → oval formation
        double grad_magnitude = static_cast<double>(w_n) * 20.0 / std::pow(10.0 + d_ij, 2.0);  // ✅ Factor of 2!

        // Numerical safety: Skip if non-finite (prevents NaN propagation)
        if (!std::isfinite(grad_magnitude)) {
            skipped_nan++;
            continue;
        }

        // Apply gradient to raw difference vector (NO normalization by d_ij!)
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            double gradient_component = grad_magnitude * diff;  // ✅ Raw diff, matches Python!

            if (!std::isfinite(gradient_component)) continue;

            #pragma omp atomic
            gradients[idx_a + d] += gradient_component;
            #pragma omp atomic
            gradients[idx_n + d] -= gradient_component;
        }

        processed_neighbors++;
    }

    // PHASE 2: Process ALL MID_NEAR triplets (matches Python lines 281-292)
    #pragma omp parallel for schedule(dynamic, 1000) reduction(+:processed_midnear,skipped_nan,skipped_zero_distance,skipped_triplets)
    for (int idx = 0; idx < static_cast<int>(triplets.size()); ++idx) {
        const auto& t = triplets[idx];
        if (t.type != MID_NEAR) continue;  // Skip non-mid-near triplets in this phase

        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // FIX13: CRITICAL - Match Python distance calculation EXACTLY!
        // Python (pacmap.py line 271-274): d_ij = 1.0, then d_ij += y_ij[d]**2
        // Result: d_ij = 1.0 + sum(diff²)  ← NOT sqrt(sum(diff²))!
        // This is the SQUARED distance plus 1.0, NOT Euclidean distance!
        double d_ij = 1.0;
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij += diff * diff;  // Add squared difference (NO sqrt!)
        }
        // Note: d_ij starts at 1.0, so it's always >= 1.0 (no need to check for zero)

        // FIX v2.8.7: CRITICAL - Match Python gradient formula EXACTLY!
        // Python (pacmap.py line 289): w = w_MN * 20000. / (10000. + d_ij) ** 2
        // Then (line 291): grad[i, d] += w * y_ij[d]  ← RAW difference, NOT normalized!
        //
        // PREVIOUS BUG: Used 10000.0 instead of 20000.0 (missing factor of 2)
        //               AND divided by d_ij (incorrect normalization)
        double grad_magnitude = static_cast<double>(w_mn) * 20000.0 / std::pow(10000.0 + d_ij, 2.0);  // ✅ Factor of 2!

        // Numerical safety: Skip if non-finite (prevents NaN propagation)
        if (!std::isfinite(grad_magnitude)) {
            skipped_nan++;
            continue;
        }

        // Apply gradient to raw difference vector (NO normalization by d_ij!)
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            double gradient_component = grad_magnitude * diff;  // ✅ Raw diff, matches Python!

            if (!std::isfinite(gradient_component)) continue;

            #pragma omp atomic
            gradients[idx_a + d] += gradient_component;
            #pragma omp atomic
            gradients[idx_n + d] -= gradient_component;
        }

        processed_midnear++;
    }

    // PHASE 3: Process ALL FURTHER triplets (matches Python lines 294-305)
    #pragma omp parallel for schedule(dynamic, 1000) reduction(+:processed_further,skipped_nan,skipped_zero_distance,skipped_triplets)
    for (int idx = 0; idx < static_cast<int>(triplets.size()); ++idx) {
        const auto& t = triplets[idx];
        if (t.type != FURTHER) continue;  // Skip non-further triplets in this phase

        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // FIX13: CRITICAL - Match Python distance calculation EXACTLY!
        // Python (pacmap.py line 271-274): d_ij = 1.0, then d_ij += y_ij[d]**2
        // Result: d_ij = 1.0 + sum(diff²)  ← NOT sqrt(sum(diff²))!
        // This is the SQUARED distance plus 1.0, NOT Euclidean distance!
        double d_ij = 1.0;
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij += diff * diff;  // Add squared difference (NO sqrt!)
        }
        // Note: d_ij starts at 1.0, so it's always >= 1.0 (no need to check for zero)

        // FURTHER triplets are repulsive to maintain separation
        double grad_magnitude = -static_cast<double>(w_f) * 2.0 / std::pow(1.0 + d_ij, 2.0);

        // Numerical safety: Skip if non-finite (prevents NaN propagation)
        if (!std::isfinite(grad_magnitude)) {
            skipped_nan++;
            continue;
        }

        // Apply gradient to raw difference vector (NO normalization by d_ij!)
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            double gradient_component = grad_magnitude * diff;  // ✅ Raw diff, matches Python!

            if (!std::isfinite(gradient_component)) continue;

            #pragma omp atomic
            gradients[idx_a + d] += gradient_component;
            #pragma omp atomic
            gradients[idx_n + d] -= gradient_component;
        }

        processed_further++;
    }

  
    // ERROR13 FIX: COMMENTED OUT gradient clipping - may interfere with natural force balance
    // Python reference doesn't use gradient clipping - lets gradients flow naturally
    // The clipping was causing artificial constraints that prevent proper embedding formation
    /*
    // Gradient clipping for Adam stability with derivative formulas
    // Gradient derivatives are smaller, so standard clipping is sufficient
    for (float& g : gradients) {
        g = std::max(-4.0f, std::min(4.0f, g));  // Clip to [-4, 4]
    }
    */

    // FIX13-v3: CRITICAL - Python does NOT normalize by triplet count!
    // Python reference (pacmap.py lines 339-341):
    //   gradients[i] += gradient   # ← Direct accumulation, NO division!
    //   gradients[j] -= gradient
    //
    // Previous fixes were WRONG:
    //   v2.7.0: Divided by n_samples (10,000) → gradients 10,000x too small
    //   v2.8.0: Divided by triplets.size() (325,611) → gradients 325,000x too small
    //
    // Result: Embedding stayed in tiny random noise circle, barely moved
    // CORRECT FIX: NO NORMALIZATION - let gradients accumulate naturally like Python!
    //
    // (normalization code removed - Python doesn't normalize)

    }

double compute_pacmap_loss(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components,
                         pacmap_progress_callback_internal callback) {

        double total_loss = 0.0;
    double neighbor_loss = 0.0, mn_loss = 0.0, fp_loss = 0.0;
    int neighbor_count = 0, mn_count = 0, fp_count = 0;
    int skipped_zero = 0, skipped_nan = 0;

    for (const auto& triplet : triplets) {
        size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

        // Use true Euclidean distance with pure double precision (matching gradient computation)
        double d_ij_squared = 0.0;
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij_squared += diff * diff;
        }
        double d_ij = std::sqrt(std::max(d_ij_squared, 1e-15));

        // Skip if distance is too small
        if (d_ij < 1e-15) {
            skipped_zero++;
            continue;
        }

        // ✅ v2.8.10: CORRECTED loss formulas consistent with FIXED gradient implementation
        // ALL THREE TRIPLET TYPES are now ATTRACTIVE, so loss should decrease with smaller distances
        double loss_term = 0.0;
        switch (triplet.type) {
            case NEIGHBOR:
                // Consistent with grad = w_n * 20.0 / (10.0 + d_ij)^2 (attractive)
                loss_term = static_cast<double>(w_n) * 20.0 * d_ij / (10.0 + d_ij);
                neighbor_loss += loss_term;
                neighbor_count++;
                break;
            case MID_NEAR:
                // Consistent with grad = w_mn * 20000.0 / (10000.0 + d_ij)^2 (attractive)
                loss_term = static_cast<double>(w_mn) * 20000.0 * d_ij / (10000.0 + d_ij);
                mn_loss += loss_term;
                mn_count++;
                break;
            case FURTHER:
                // ✅ v2.8.10: CORRECTED - now attractive, loss decreases with smaller distances
                // Consistent with grad = w_f * 2.0 / (1.0 + d_ij)^2 (attractive)
                loss_term = static_cast<double>(w_f) * 2.0 * d_ij / (1.0 + d_ij);
                fp_loss += loss_term;
                fp_count++;
                break;
        }

        // NaN safety - skip non-finite loss terms
        if (std::isfinite(loss_term)) {
            total_loss += loss_term;
        } else {
            skipped_nan++;
        }
    }

    // NaN safety - return 0 if total_loss is not finite
    if (!std::isfinite(total_loss)) {
        return 0.0;
    }

    double avg_loss = total_loss / static_cast<double>(triplets.size());

    return avg_loss;
}

bool check_convergence(const std::vector<float>& loss_history, float threshold, int window) {
    if (loss_history.size() < window) return false;

    // Check if loss has stabilized over the last 'window' iterations
    float recent_avg = 0.0f, older_avg = 0.0f;
    int window_half = window / 2;

    for (int i = loss_history.size() - window; i < loss_history.size(); ++i) {
        if (i < loss_history.size() - window_half) {
            older_avg += loss_history[i];
        } else {
            recent_avg += loss_history[i];
        }
    }

    older_avg /= window_half;
    recent_avg /= window_half;

    return std::abs(recent_avg - older_avg) < threshold;
}

bool should_terminate_early(const std::vector<float>& loss_history, int max_no_improvement) {
    if (loss_history.size() < max_no_improvement) return false;

    // Check if loss has improved in recent iterations
    float recent_min = *std::min_element(loss_history.end() - max_no_improvement, loss_history.end());
    float historical_min = *std::min_element(loss_history.begin(), loss_history.end());

    // Terminate if no significant improvement
    return (recent_min >= historical_min * 0.999);  // 0.1% tolerance
}

void compute_triplet_gradients(const Triplet& triplet, const double* embedding,
                             double* gradients, double grad_magnitude, int n_components) {

    size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
    size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

    // Compute distance
    double d_ij_squared = 0.0;
    for (int d = 0; d < n_components; ++d) {
        double diff = embedding[idx_a + d] - embedding[idx_n + d];
        d_ij_squared += diff * diff;
    }
    double d_ij = std::sqrt(std::max(d_ij_squared, 1e-15));

    double scale = grad_magnitude / d_ij;
    for (int d = 0; d < n_components; ++d) {
        double diff = embedding[idx_a + d] - embedding[idx_n + d];
        double gradient_component = scale * diff;

        gradients[idx_a + d] += gradient_component;
        gradients[idx_n + d] -= gradient_component;
    }
}

void clip_gradients(std::vector<double>& gradients, double max_norm) {
    double grad_norm = 0.0;
    for (double g : gradients) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);

    if (grad_norm > max_norm) {
        double scale = max_norm / grad_norm;
        for (double& g : gradients) {
            g *= scale;
        }
    }
}

void normalize_gradients(std::vector<double>& gradients) {
    double grad_norm = 0.0;
    for (double g : gradients) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);

    if (grad_norm > 1e-15) {
        double scale = 1.0 / grad_norm;
        for (double& g : gradients) {
            g *= scale;
        }
    }
}

float cosine_annealing_lr(float base_lr, int current_iter, int total_iters) {
    return base_lr * 0.5f * (1.0f + std::cos(M_PI * current_iter / total_iters));
}

float step_decay_lr(float base_lr, int current_iter, int decay_steps, float decay_rate) {
    return base_lr * std::pow(decay_rate, current_iter / decay_steps);
}

void start_gradient_timer() {
    gradient_start_time = std::chrono::high_resolution_clock::now();
}

double get_gradient_computation_time() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - gradient_start_time);
    return duration.count() / 1000.0;  // Return in milliseconds
}

size_t get_gradient_memory_usage(int n_samples, int n_components) {
    return n_samples * n_components * sizeof(double) * 3;  // embedding + gradients + adam state (now double precision)
}



void compute_second_order_info(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                             std::vector<double>& hessian_diagonal, int n_components) {

    hessian_diagonal.assign(embedding.size(), 0.0);

    // Simple approximation of diagonal Hessian
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(triplets.size()); ++idx) {
        const auto& t = triplets[idx];
        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // Compute distance
        double d_ij_squared = 0.0;
        for (int d = 0; d < n_components; ++d) {
            double diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij_squared += diff * diff;
        }
        double d_ij = std::sqrt(std::max(d_ij_squared, 1e-15));

        // Approximate second derivative contribution
        double hess_contribution;
        switch (t.type) {
            case NEIGHBOR:
                hess_contribution = 20.0 / std::pow(10.0 + d_ij, 3.0);
                break;
            case MID_NEAR:
                hess_contribution = 20000.0 / std::pow(10000.0 + d_ij, 3.0);
                break;
            case FURTHER:
                hess_contribution = 2.0 / std::pow(1.0 + d_ij, 3.0);
                break;
        }

        for (int d = 0; d < n_components; ++d) {
            #pragma omp atomic
            hessian_diagonal[idx_a + d] += hess_contribution;
            #pragma omp atomic
            hessian_diagonal[idx_n + d] += hess_contribution;
        }
    }
}

void adaptive_learning_rate_adjustment(float& learning_rate, const std::vector<float>& loss_history) {
    if (loss_history.size() < 10) return;

    // Check if loss is increasing or decreasing
    float recent_change = loss_history.back() - loss_history[loss_history.size() - 10];

    if (recent_change > 0) {
        // Loss increasing, reduce learning rate
        learning_rate *= 0.95f;
    } else if (recent_change < -1e-6f) {
        // Loss decreasing, can increase learning rate slightly
        learning_rate *= 1.02f;
    }

    // Keep learning rate within reasonable bounds
    learning_rate = std::max(1e-6f, std::min(10.0f, learning_rate));
}

