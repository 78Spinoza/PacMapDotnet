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
    // CRITICAL FIX: Use actual phase boundaries instead of hardcoded values (errors6.txt)
    // Previous implementation used hardcoded 100/200 which didn't match model settings!
    float w_n, w_mn, w_f = 1.0f;

    if (current_iter < phase1_end) {
        // Phase 1: Global structure (user-defined iterations)
        w_n = 3.0f;
        w_mn = 3.0f;
    } else if (current_iter < phase2_end) {
        // Phase 2: Balance phase (user-defined iterations)
        w_n = 3.0f;
        w_mn = 1.0f;
    } else {
        // Phase 3: Local structure (remaining iterations)
        // CRITICAL FIX: Maintain repulsion to prevent collapse!
        w_n = 1.0f;   // Increased from 0.1f
        w_mn = 1.0f;  // Changed from 0.0f to prevent collapse
    }

    return {w_n, w_mn, w_f};
}

void compute_gradients(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                       std::vector<float>& gradients, float w_n, float w_mn, float w_f, int n_components) {

    gradients.assign(embedding.size(), 0.0f);

    // CRITICAL FIX: Updated gradient formulas from error5.txt
    float total_coeff = 0.0f;
    int valid_triplets = 0;
    float min_dist = std::numeric_limits<float>::max();
    float max_dist = std::numeric_limits<float>::lowest();

    #pragma omp parallel for schedule(dynamic, 1000) reduction(+:total_coeff, valid_triplets) reduction(min:min_dist) reduction(max:max_dist)
    for (int idx = 0; idx < static_cast<int>(triplets.size()); ++idx) {
        const auto& t = triplets[idx];
        if (t.anchor == t.neighbor) continue;
        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        float dist_squared = 0.0f;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            dist_squared += diff * diff;
        }
        dist_squared = std::max(dist_squared, 1e-8f);
        float dist = std::sqrt(dist_squared);

        // Updated gradient formulas from error5.txt
        float coeff = 0.0f;
        switch (t.type) {
            case NEIGHBOR:
                coeff = -w_n * 20.0f / std::pow(10.0f + dist_squared, 2.0f);
                break;
            case MID_NEAR:
                coeff = -w_mn * 2.0f / std::pow(1.0f + dist_squared, 2.0f);
                break;
            case FURTHER:
                coeff = w_f * 6.0f / std::pow(1.0f + dist_squared, 2.0f);
                break;
            default: continue;
        }

        total_coeff += std::abs(coeff);
        valid_triplets++;
        min_dist = std::min(min_dist, dist);
        max_dist = std::max(max_dist, dist);

        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            float gradient_component = coeff * diff;

            #pragma omp atomic
            gradients[idx_a + d] += gradient_component;
            #pragma omp atomic
            gradients[idx_n + d] -= gradient_component;
        }
    }

    // Debug gradient statistics
    printf("[DEBUG] GRADIENT: Valid triplets=%d, Avg coeff=%.6f, Distance range=[%.6f, %.6f]\n",
           valid_triplets, valid_triplets ? total_coeff / valid_triplets : 0, min_dist, max_dist);
}

void adagrad_update(std::vector<float>& embedding, const std::vector<float>& gradients,
                  std::vector<float>& m, std::vector<float>& v, int iter, float learning_rate,
                  float beta1, float beta2, float eps) {

    // FIXED: AdaGrad optimizer - use model->learning_rate instead of hardcoded 1.0f
    // This was the critical bug causing poor PACMAP performance!
    const float ada_grad_lr = learning_rate;  // Use the passed learning_rate parameter
    const float ada_grad_eps = 1e-8f;

    // Parallel AdaGrad update - simple accumulation of squared gradients
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(embedding.size()); ++i) {
        // Accumulate squared gradients (no momentum, no bias correction)
        v[i] += gradients[i] * gradients[i];

        // Update parameters with adaptive learning rate using proper learning_rate
        embedding[i] -= ada_grad_lr * gradients[i] / (std::sqrt(v[i]) + ada_grad_eps);
    }
}

void initialize_adagrad_state(std::vector<float>& m, std::vector<float>& v, size_t size) {
    m.assign(size, 0.0f);
    v.assign(size, 0.0f);
}

void reset_adagrad_state(std::vector<float>& m, std::vector<float>& v) {
    std::fill(m.begin(), m.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
}

float compute_pacmap_loss(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components) {

    // CRITICAL FIX: Updated loss function from error5.txt
    float total_loss = 0.0f;
    int count = 0;

    for (const auto& triplet : triplets) {
        size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

        float dist_squared = 0.0f;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            dist_squared += diff * diff;
        }
        float dist = std::sqrt(std::max(dist_squared, 1e-8f));

        float loss_term = 0.0f;
        switch (triplet.type) {
            case NEIGHBOR:
                loss_term = w_n * 10.0f * std::log1p(dist);
                break;
            case MID_NEAR:
                loss_term = w_mn * std::log1p(dist);
                break;
            case FURTHER:
                loss_term = w_f / (1.0f + dist_squared);
                break;
            default: continue;
        }
        total_loss += loss_term;
        count++;
    }

    printf("[DEBUG] LOSS: Total=%.6f, Avg=%.6f\n", total_loss, count ? total_loss / count : 0);
    return total_loss;
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

void apply_adagrad_params(AdaGradParams& params, float learning_rate,
                         float beta1, float beta2, float eps) {
    params.learning_rate = learning_rate;
    params.beta1 = beta1;     // Note: Not used in AdaGrad, kept for compatibility
    params.beta2 = beta2;     // Note: Not used in AdaGrad, kept for compatibility
    params.epsilon = eps;
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

void print_adagrad_state(const std::vector<float>& m, const std::vector<float>& v, int iter) {
    if (m.empty() || v.empty()) return;

    float m_mean = 0.0f, v_mean = 0.0f;
    for (size_t i = 0; i < m.size(); ++i) {
        m_mean += m[i];
        v_mean += v[i];
    }
    m_mean /= m.size();
    v_mean /= v.size();

    std::cout << "AdaGrad State (iter " << iter << "):" << std::endl;
    std::cout << "  First moment mean: " << m_mean << " (Note: Not used in AdaGrad)" << std::endl;
    std::cout << "  Second moment mean: " << v_mean << " (Squared gradients accumulator)" << std::endl;
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