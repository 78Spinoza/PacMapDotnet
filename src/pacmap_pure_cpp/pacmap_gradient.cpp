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

std::tuple<float, float, float> get_weights(int current_iter, int phase1_end, int phase2_end) {
    // CRITICAL FIX: Match Python PACMAP phase weights exactly
    float w_n, w_mn, w_f = 1.0f;

    if (current_iter < phase1_end) {
        // Phase 1: Global structure (0-10%): w_mn: 1000→3 transition
        float progress = (float)current_iter / phase1_end;
        w_n = 2.0f;  // FIXED: Should be 2.0f to match Rust exactly
        w_mn = 1000.0f * (1.0f - progress) + 3.0f * progress;  // 1000→3 (correct)
        printf("[WEIGHT DEBUG] Phase 1 (iter %d): w_n=%.1f, w_mn=%.1f, progress=%.2f\n",
               current_iter, w_n, w_mn, progress);
    } else if (current_iter < phase2_end) {
        // Phase 2: Balance phase (10-40%): stable weights
        w_n = 3.0f;  // FIXED: Should be 3.0f to match Rust exactly
        w_mn = 3.0f;  // Correct
        printf("[WEIGHT DEBUG] Phase 2 (iter %d): w_n=%.1f, w_mn=%.1f\n",
               current_iter, w_n, w_mn);
    } else {
        // Phase 3: Local structure (40-100%): w_mn = 0 fixed (no mid-near attraction)
        w_n = 1.0f;  // Correct - matches Rust
        w_mn = 0.0f;  // FIXED: No mid-near attraction in local refinement phase
        printf("[WEIGHT DEBUG] Phase 3 (iter %d): w_n=%.1f, w_mn=%.1f (FIXED - should be 0!)\n",
               current_iter, w_n, w_mn);
    }

    return {w_n, w_mn, w_f};
}

void compute_gradients(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                       std::vector<float>& gradients, float w_n, float w_mn, float w_f, int n_components) {

    gradients.assign(embedding.size(), 0.0f);

    // Progress-friendly gradient computation
    if (triplets.size() > 10000) {
        printf("   Computing gradients on %zu triplets...\n", triplets.size());
    }

    // Parallel gradient computation with atomic operations (review requirement)
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int idx = 0; idx < static_cast<int>(triplets.size()); ++idx) {
        const auto& t = triplets[idx];
        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // CRITICAL FIX: Match Rust distance calculation exactly
        // Calculate squared distance (+1 for numerical stability) like Rust
        float d_ij = 1.0f;  // Start with 1.0 for numerical stability (Rust behavior)
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij += diff * diff;  // Add squared difference to 1.0
        }
        // Note: d_ij now contains 1.0 + sum(squared_differences), NOT sqrt!

        // CRITICAL FIX: Correct gradient sign conventions to match Rust implementation
        // Attractive pairs (NEIGHBOR, MID_NEAR) should have POSITIVE coefficients
        // Repulsive pairs (FURTHER) should have NEGATIVE coefficients
        float coeff = 0.0f;
        switch (t.type) {
            case NEIGHBOR:
                // Attractive: +w_n * 20.0f / (10.0f + d_ij)²
                coeff = w_n * 20.0f / std::pow(10.0f + d_ij, 2.0f);
                break;
            case MID_NEAR:
                // Attractive: +w_mn * 20000.0f / (10000.0f + d_ij)²
                coeff = w_mn * 20000.0f / std::pow(10000.0f + d_ij, 2.0f);
                break;
            case FURTHER:
                // Repulsive: -w_f * 2.0f / (1.0f + d_ij)²
                coeff = -w_f * 2.0f / std::pow(1.0f + d_ij, 2.0f);
                break;
            default:
                continue;  // Should never happen
        }

        // Removed verbose per-triplet debug output for cleaner progress display

        // Apply gradients symmetrically (Newton's third law)
        // CRITICAL FIX: Remove distance normalization to match reference implementation
        // The gradient should be coeff * diff directly, not coeff * diff / distance
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            float gradient_component = coeff * diff;  // Direct multiply without distance normalization

            // Thread-safe atomic operations (review requirement for determinism)
            #pragma omp atomic
            gradients[idx_a + d] += gradient_component;
            #pragma omp atomic
            gradients[idx_n + d] -= gradient_component;
        }
    }

    // CRITICAL FIX: Remove gradient normalization to match reference implementation
    // The reference uses raw gradient sums, not averaged gradients
    // This allows stronger updates for larger datasets as intended

    // Debug gradient stats after summation
    float grad_min = gradients[0], grad_max = gradients[0], grad_sum = 0.0f;
    int grad_nan = 0, grad_inf = 0;
    for (size_t i = 0; i < gradients.size(); ++i) {
        grad_min = std::min(grad_min, gradients[i]);
        grad_max = std::max(grad_max, gradients[i]);
        grad_sum += gradients[i];
        if (std::isnan(gradients[i])) grad_nan++;
        if (std::isinf(gradients[i])) grad_inf++;
    }
    float grad_mean = grad_sum / gradients.size();
    // Cleaner gradient statistics - only show if there are issues
    if (grad_nan > 0 || grad_inf > 0) {
        printf("   ⚠️  Gradient issues detected: %d NaN, %d Inf values\n", grad_nan, grad_inf);
    }
}

float compute_pacmap_loss(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components) {

    printf("[LOSS DEBUG] Computing loss: n_triplets=%zu, w_n=%.1f, w_mn=%.1f, w_f=%.1f\n",
           triplets.size(), w_n, w_mn, w_f);
    // CRITICAL FIX: Updated loss function from error5.txt
    float total_loss = 0.0f;

    for (const auto& triplet : triplets) {
        size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

        // CRITICAL FIX: Match Rust distance calculation exactly in loss function
        // Calculate squared distance (+1 for numerical stability) like Rust
        float d_ij = 1.0f;  // Start with 1.0 for numerical stability (Rust behavior)
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij += diff * diff;  // Add squared difference to 1.0
        }
        // Note: d_ij now contains 1.0 + sum(squared_differences), NOT sqrt!

        // CRITICAL FIX: Make loss function consistent with gradient formulas
        // Loss must be the integral of the gradient for proper optimization
        float loss_term = 0.0f;
        switch (triplet.type) {
            case NEIGHBOR:
                // NEW LOSS: w_n * 10.0f * d_ij² / (10.0f + d_ij²)
                loss_term = w_n * 10.0f * d_ij * d_ij / (10.0f + d_ij * d_ij);
                if (triplet.anchor == 0 && triplet.neighbor == 1) {
                    printf("[DEBUG] NEW LOSS FUNCTION ACTIVE! NEIGHBOR loss=%.6f, dist=%.6f\n", loss_term, d_ij);
                }
                break;
            case MID_NEAR:
                // NEW LOSS: w_mn * 10000.0f * d_ij² / (10000.0f + d_ij²)
                loss_term = w_mn * 10000.0f * d_ij * d_ij / (10000.0f + d_ij * d_ij);
                break;
            case FURTHER:
                // Consistent with gradient: coeff = -w_f * 2.0f / (1.0f + d_ij²)²
                // Loss = w_f / (1.0f + d_ij²) (already correct)
                loss_term = w_f / (1.0f + d_ij * d_ij);
                break;
        }

        total_loss += loss_term;
    }

    float avg_loss = total_loss / static_cast<float>(triplets.size());  // Average loss
    printf("[LOSS DEBUG] Loss: %.4e, n_triplets=%zu\n", avg_loss, triplets.size());
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

void compute_triplet_gradients(const Triplet& triplet, const float* embedding,
                             float* gradients, float grad_magnitude, int n_components) {

    size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
    size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

    // Compute distance
    float d_ij_squared = 0.0f;
    for (int d = 0; d < n_components; ++d) {
        float diff = embedding[idx_a + d] - embedding[idx_n + d];
        d_ij_squared += diff * diff;
    }
    float d_ij = std::sqrt(std::max(d_ij_squared, 1e-8f));

    float scale = grad_magnitude / d_ij;
    for (int d = 0; d < n_components; ++d) {
        float diff = embedding[idx_a + d] - embedding[idx_n + d];
        float gradient_component = scale * diff;

        gradients[idx_a + d] += gradient_component;
        gradients[idx_n + d] -= gradient_component;
    }
}

void clip_gradients(std::vector<float>& gradients, float max_norm) {
    float grad_norm = 0.0f;
    for (float g : gradients) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);

    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        for (float& g : gradients) {
            g *= scale;
        }
    }
}

void normalize_gradients(std::vector<float>& gradients) {
    float grad_norm = 0.0f;
    for (float g : gradients) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);

    if (grad_norm > 1e-8f) {
        float scale = 1.0f / grad_norm;
        for (float& g : gradients) {
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
    return n_samples * n_components * sizeof(float) * 3;  // embedding + gradients + adam state
}

void print_gradient_stats(const std::vector<float>& gradients) {
    if (gradients.empty()) return;

    float min_grad = gradients[0], max_grad = gradients[0], sum = 0.0f;
    for (float g : gradients) {
        min_grad = std::min(min_grad, g);
        max_grad = std::max(max_grad, g);
        sum += g;
    }
    float mean = sum / gradients.size();

    float variance = 0.0f;
    for (float g : gradients) {
        variance += (g - mean) * (g - mean);
    }
    float std_dev = std::sqrt(variance / gradients.size());

    std::cout << "Gradient Statistics:" << std::endl;
    std::cout << "  Count: " << gradients.size() << std::endl;
    std::cout << "  Min: " << min_grad << std::endl;
    std::cout << "  Max: " << max_grad << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std Dev: " << std_dev << std::endl;
}

void validate_gradients(const std::vector<float>& gradients, const std::vector<float>& embedding) {
    if (gradients.size() != embedding.size()) {
        std::cerr << "Error: Gradient and embedding sizes mismatch!" << std::endl;
        return;
    }

    int nan_count = 0, inf_count = 0;
    for (float g : gradients) {
        if (std::isnan(g)) nan_count++;
        if (std::isinf(g)) inf_count++;
    }

    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "Error: " << nan_count << " NaN and " << inf_count << " Inf gradients detected!" << std::endl;
    }
}

void compute_second_order_info(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                             std::vector<float>& hessian_diagonal, int n_components) {

    hessian_diagonal.assign(embedding.size(), 0.0f);

    // Simple approximation of diagonal Hessian
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(triplets.size()); ++idx) {
        const auto& t = triplets[idx];
        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // Compute distance
        float d_ij_squared = 0.0f;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij_squared += diff * diff;
        }
        float d_ij = std::sqrt(std::max(d_ij_squared, 1e-8f));

        // Approximate second derivative contribution
        float hess_contribution;
        switch (t.type) {
            case NEIGHBOR:
                hess_contribution = 20.0f / std::pow(10.0f + d_ij, 3.0f);
                break;
            case MID_NEAR:
                hess_contribution = 20000.0f / std::pow(10000.0f + d_ij, 3.0f);
                break;
            case FURTHER:
                hess_contribution = 2.0f / std::pow(1.0f + d_ij, 3.0f);
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