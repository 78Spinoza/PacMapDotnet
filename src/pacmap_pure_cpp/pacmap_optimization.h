#pragma once

#include "pacmap_model.h"
#include "pacmap_gradient.h"
#include <vector>

// Main optimization function with Adam and three-phase weights
extern void optimize_embedding(PacMapModel* model, float* embedding_out, uwot_progress_callback_v2 callback);

// Initialization utilities
extern void initialize_random_embedding(std::vector<float>& embedding, int n_samples, int n_components, std::mt19937& rng);
extern void initialize_adam_optimization(PacMapModel* model, std::vector<float>& m, std::vector<float>& v);

// Optimization phases
extern void run_optimization_phase(PacMapModel* model, std::vector<float>& embedding,
                                  std::vector<float>& m, std::vector<float>& v,
                                  int start_iter, int end_iter, const std::string& phase_name,
                                  uwot_progress_callback_v2 callback);

// Three-phase optimization control
struct OptimizationPhase {
    std::string name;
    int start_iter;
    int end_iter;
    float target_weight_mn;
    float target_weight_n;
    float target_weight_f;
};

extern std::vector<OptimizationPhase> get_optimization_phases(int phase1_iters, int phase2_iters, int phase3_iters);

// Safety and monitoring
extern void compute_safety_stats(PacMapModel* model, const std::vector<float>& embedding);
extern void monitor_optimization_progress(int iter, int total_iters, float loss,
                                         const std::string& phase, uwot_progress_callback_v2 callback);

// Convergence and quality metrics
extern float compute_embedding_quality(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                                     int n_components);
extern bool check_optimization_convergence(const std::vector<float>& loss_history, float tolerance = 1e-6);

// Optimization strategies
enum class OptimizationStrategy {
    STANDARD,           // Standard three-phase PACMAP optimization
    FAST,              // Faster convergence with aggressive learning rates
    ACCURATE,          // Higher precision with more iterations
    MEMORY_EFFICIENT   // Optimized for large datasets
};

extern void apply_optimization_strategy(PacMapModel* model, OptimizationStrategy strategy);

// Advanced optimization features
extern void adaptive_phase_transitions(PacMapModel* model, std::vector<float>& loss_history);
extern void dynamic_learning_rate_adjustment(PacMapModel* model, int iter, float current_loss);
extern void early_stopping_check(PacMapModel* model, const std::vector<float>& loss_history, int patience = 50);

// Optimization diagnostics
struct OptimizationDiagnostics {
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    float loss_reduction = 0.0f;
    int total_iterations = 0;
    double optimization_time_ms = 0.0f;
    bool converged = false;
    int convergence_iteration = -1;
    std::vector<float> loss_history;
};

extern OptimizationDiagnostics run_optimization_with_diagnostics(PacMapModel* model,
                                                               std::vector<float>& embedding,
                                                               uwot_progress_callback_v2 callback);

// Phase detection utilities
extern std::string get_current_phase(int iter, int phase1_iters, int phase2_iters);
extern float get_phase_weight(int iter, int total_iters, float start_weight, float end_weight);

// Quality assessment
extern void assess_embedding_quality(const std::vector<float>& embedding, const PacMapModel* model);
extern float compute_trustworthiness(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                                     int n_samples, int n_components);

// Performance optimization
extern void optimize_for_dataset_size(PacMapModel* model, int n_samples);
extern void configure_memory_usage(PacMapModel* model, size_t available_memory_mb);

// Debug and validation
extern void validate_optimization_state(const PacMapModel* model, const std::vector<float>& embedding);
extern void print_optimization_summary(const OptimizationDiagnostics& diagnostics);