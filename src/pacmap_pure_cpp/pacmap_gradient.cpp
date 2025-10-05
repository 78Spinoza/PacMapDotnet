#include "pacmap_gradient.h"
#include "pacmap_distance.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <chrono>

static std::chrono::high_resolution_clock::time_point gradient_start_time;

std::tuple<float, float, float> get_weights(int current_iter, int total_iters) {
    float progress = static_cast<float>(current_iter) / total_iters;
    float w_n = 1.0f, w_f = 1.0f, w_mn;

    // Three-phase weight schedule (review specification)
    if (progress < 0.1f) {
        // Phase 1: Global structure focus (0-10%)
        // w_MN decreases linearly from 1000 to 3
        float phase_progress = progress * 10.0f;  // 0 to 1 within phase
        w_mn = 1000.0f * (1.0f - phase_progress) + 3.0f * phase_progress;
    } else if (progress < 0.4f) {
        // Phase 2: Balance phase (10-40%)
        w_mn = 3.0f;
    } else {
        // Phase 3: Local structure focus (40-100%)
        // w_MN decreases linearly from 3 to 0
        float phase_progress = (progress - 0.4f) / 0.6f;  // 0 to 1 within phase
        w_mn = 3.0f * (1.0f - phase_progress);
    }

    return {w_n, w_mn, w_f};
}

void compute_gradients(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                       std::vector<float>& gradients, float w_n, float w_mn, float w_f, int n_components) {

    gradients.assign(embedding.size(), 0.0f);

    // Parallel gradient computation with atomic operations (review requirement)
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t idx = 0; idx < triplets.size(); ++idx) {
        const auto& t = triplets[idx];
        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // Compute Euclidean distance in embedding space
        float d_ij_squared = 0.0f;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij_squared += diff * diff;
        }
        float d_ij = std::sqrt(std::max(d_ij_squared, 1e-8f));

        // Compute gradient magnitude based on triplet type (PACMAP loss functions)
        float grad_magnitude;
        switch (t.type) {
            case NEIGHBOR:
                // Attractive force: w * 10 / ((10 + d)^2)
                grad_magnitude = w_n * 10.0f / std::pow(10.0f + d_ij, 2.0f);
                break;
            case MID_NEAR:
                // Moderate attractive force: w * 10000 / ((10000 + d)^2)
                grad_magnitude = w_mn * 10000.0f / std::pow(10000.0f + d_ij, 2.0f);
                break;
            case FURTHER:
                // Repulsive force: -w / ((1 + d)^2)
                grad_magnitude = -w_f / std::pow(1.0f + d_ij, 2.0f);
                break;
            default:
                continue;  // Should never happen
        }

        // Apply gradients symmetrically (Newton's third law)
        float scale = grad_magnitude / d_ij;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            float gradient_component = scale * diff;

            // Thread-safe atomic operations (review requirement for determinism)
            #pragma omp atomic
            gradients[idx_a + d] += gradient_component;
            #pragma omp atomic
            gradients[idx_n + d] -= gradient_component;
        }
    }
}

void adam_update(std::vector<float>& embedding, const std::vector<float>& gradients,
                 std::vector<float>& m, std::vector<float>& v, int iter, float learning_rate,
                 float beta1, float beta2, float eps) {

    // Bias correction terms (Adam algorithm)
    float beta1_pow = std::pow(beta1, iter + 1);
    float beta2_pow = std::pow(beta2, iter + 1);

    // Parallel Adam update with adaptive learning rates
    #pragma omp parallel for
    for (size_t i = 0; i < embedding.size(); ++i) {
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];

        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * (gradients[i] * gradients[i]);

        // Compute bias-corrected estimates
        float m_hat = m[i] / (1 - beta1_pow);
        float v_hat = v[i] / (1 - beta2_pow);

        // Update parameters with adaptive learning rate
        embedding[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void initialize_adam_state(std::vector<float>& m, std::vector<float>& v, size_t size) {
    m.assign(size, 0.0f);
    v.assign(size, 0.0f);
}

void reset_adam_state(std::vector<float>& m, std::vector<float>& v) {
    std::fill(m.begin(), m.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
}

float compute_pacmap_loss(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components) {

    float total_loss = 0.0f;

    for (const auto& triplet : triplets) {
        size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

        // Compute embedding space distance
        float d_ij = compute_sampling_distance(embedding.data() + idx_a,
                                             embedding.data() + idx_n,
                                             n_components, PACMAP_METRIC_EUCLIDEAN);

        // Compute loss based on triplet type
        float triplet_loss;
        switch (triplet.type) {
            case NEIGHBOR:
                triplet_loss = w_n * (d_ij / (10.0f + d_ij));
                break;
            case MID_NEAR:
                triplet_loss = w_mn * (d_ij / (10000.0f + d_ij));
                break;
            case FURTHER:
                triplet_loss = w_f * (1.0f / (1.0f + d_ij));
                break;
        }

        total_loss += triplet_loss;
    }

    return total_loss / static_cast<float>(triplets.size());  // Average loss
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

void apply_adam_params(AdamParams& params, float learning_rate,
                     float beta1, float beta2, float eps) {
    params.learning_rate = learning_rate;
    params.beta1 = beta1;
    params.beta2 = beta2;
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

void print_adam_state(const std::vector<float>& m, const std::vector<float>& v, int iter) {
    if (m.empty() || v.empty()) return;

    float m_mean = 0.0f, v_mean = 0.0f;
    for (size_t i = 0; i < m.size(); ++i) {
        m_mean += m[i];
        v_mean += v[i];
    }
    m_mean /= m.size();
    v_mean /= v.size();

    std::cout << "Adam State (iter " << iter << "):" << std::endl;
    std::cout << "  First moment mean: " << m_mean << std::endl;
    std::cout << "  Second moment mean: " << v_mean << std::endl;
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
    for (size_t idx = 0; idx < triplets.size(); ++idx) {
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