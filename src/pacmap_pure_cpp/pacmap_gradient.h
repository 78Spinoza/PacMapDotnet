#pragma once

#include "pacmap_model.h"
#include <vector>
#include <tuple>

// Forward declaration for callback type
typedef void (*pacmap_progress_callback_internal)(const char* phase, int current, int total, float percent, const char* message);

// ERROR13 FIX: Three-phase weight schedule using Python-matching progress
extern std::tuple<float, float, float> get_weights(int current_iter, int total_iters);

// 🔬 PHASE 4: Enhanced phase transition debugging with phase information
extern std::tuple<float, float, float, std::string> get_weights_with_phase_info(int current_iter, int total_iters);

// Double precision gradient computation for numerical stability (Python matching)
extern void compute_gradients(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                             std::vector<double>& gradients, float w_n, float w_mn, float w_f, int n_components,
                             pacmap_progress_callback_internal callback = nullptr);

// REMOVED: AdaGrad functions - now using Adam optimizer in main optimization loop

// Double precision loss computation for monitoring convergence (with enhanced debugging)
extern double compute_pacmap_loss(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                                float w_n, float w_mn, float w_f, int n_components,
                                pacmap_progress_callback_internal callback = nullptr);

// Convergence detection and early stopping
extern bool check_convergence(const std::vector<float>& loss_history, float threshold = 1e-6, int window = 50);
extern bool should_terminate_early(const std::vector<float>& loss_history, int max_no_improvement = 100);

// Double precision gradient computation variants
extern void compute_triplet_gradients(const Triplet& triplet, const double* embedding,
                                     double* gradients, double grad_magnitude, int n_components);

// REMOVED: AdaGrad parameter structure - now using Adam optimizer in main optimization loop

// Double precision gradient clipping and normalization
extern void clip_gradients(std::vector<double>& gradients, double max_norm);
extern void normalize_gradients(std::vector<double>& gradients);

// Learning rate scheduling
extern float cosine_annealing_lr(float base_lr, int current_iter, int total_iters);
extern float step_decay_lr(float base_lr, int current_iter, int decay_steps, float decay_rate);

// Performance monitoring
extern void start_gradient_timer();
extern double get_gradient_computation_time();
extern size_t get_gradient_memory_usage(int n_samples, int n_components);

// Double precision debug utilities
extern void print_gradient_stats(int processed_neighbors, int processed_midnear, int processed_further,
                                int skipped_nan, int skipped_zero_distance, int skipped_triplets,
                                const std::string& current_phase_name, int current_iter, int total_iters,
                                float w_n, float w_mn, float w_f);
extern bool validate_gradients(const std::vector<double>& gradients, const std::string& context);

// Double precision advanced gradient features
extern void compute_second_order_info(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                                     std::vector<double>& hessian_diagonal, int n_components);
extern void adaptive_learning_rate_adjustment(float& learning_rate, const std::vector<float>& loss_history);