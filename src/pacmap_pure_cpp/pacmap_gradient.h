#pragma once

#include "pacmap_model.h"
#include <vector>
#include <tuple>

// Three-phase weight schedule
extern std::tuple<float, float, float> get_weights(int current_iter, int total_iters);

// Parallel gradient computation with atomic operations
extern void compute_gradients(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                             std::vector<float>& gradients, float w_n, float w_mn, float w_f, int n_components);

// AdaGrad optimizer (FIXED: Replaced ADAM to match official PACMAP)
extern void adagrad_update(std::vector<float>& embedding, const std::vector<float>& gradients,
                         std::vector<float>& m, std::vector<float>& v, int iter, float learning_rate,
                         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

// AdaGrad state management (simplified - no first moment, only squared gradients)
extern void initialize_adagrad_state(std::vector<float>& m, std::vector<float>& v, size_t size);
extern void reset_adagrad_state(std::vector<float>& m, std::vector<float>& v);

// Loss computation for monitoring convergence
extern float compute_pacmap_loss(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                                float w_n, float w_mn, float w_f, int n_components);

// Convergence detection and early stopping
extern bool check_convergence(const std::vector<float>& loss_history, float threshold = 1e-6, int window = 50);
extern bool should_terminate_early(const std::vector<float>& loss_history, int max_no_improvement = 100);

// Gradient computation variants
extern void compute_triplet_gradients(const Triplet& triplet, const float* embedding,
                                     float* gradients, float grad_magnitude, int n_components);

// AdaGrad hyperparameter utilities
struct AdaGradParams {
    float beta1 = 0.9f;     // Note: Not used in AdaGrad, kept for compatibility
    float beta2 = 0.999f;   // Note: Not used in AdaGrad, kept for compatibility
    float epsilon = 1e-8f;
    float learning_rate = 1.0f;
    float weight_decay = 0.0f;
    bool amsgrad = false;   // Note: Not used in AdaGrad, kept for compatibility
};

extern void apply_adagrad_params(AdaGradParams& params, float learning_rate,
                               float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

// Gradient clipping and normalization
extern void clip_gradients(std::vector<float>& gradients, float max_norm);
extern void normalize_gradients(std::vector<float>& gradients);

// Learning rate scheduling
extern float cosine_annealing_lr(float base_lr, int current_iter, int total_iters);
extern float step_decay_lr(float base_lr, int current_iter, int decay_steps, float decay_rate);

// Performance monitoring
extern void start_gradient_timer();
extern double get_gradient_computation_time();
extern size_t get_gradient_memory_usage(int n_samples, int n_components);

// Debug utilities
extern void print_gradient_stats(const std::vector<float>& gradients);
extern void print_adagrad_state(const std::vector<float>& m, const std::vector<float>& v, int iter);
extern void validate_gradients(const std::vector<float>& gradients, const std::vector<float>& embedding);

// Advanced gradient features
extern void compute_second_order_info(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                                     std::vector<float>& hessian_diagonal, int n_components);
extern void adaptive_learning_rate_adjustment(float& learning_rate, const std::vector<float>& loss_history);