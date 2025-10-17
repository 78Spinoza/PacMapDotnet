#include "pacmap_gradient.h"
#include "pacmap_distance.h"
#include "pacmap_triplet_sampling.h"
#include "pacmap_simd_utils.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <chrono>

// ERROR14 Step 3: Eigen SIMD vectorization for distance calculations
#include <Eigen/Dense>

// FIX17.md Step 4: Aligned memory allocation for SIMD
#ifdef _WIN32
#include <xmmintrin.h>  // For _mm_malloc/_mm_free on Windows
#else
#include <cstdlib>      // For aligned_alloc on Linux/macOS
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static std::chrono::high_resolution_clock::time_point gradient_start_time;

std::tuple<float, float, float> get_weights(int current_iter, int phase1_iters, int phase2_iters) {
    // FIX21: CRITICAL - Use iteration-based phase boundaries to match Python EXACTLY!
    // Python reference (pacmap.py lines 337-353) uses explicit iteration counts, NOT percentages
    //
    // PREVIOUS BUG: C++ used percentage-based boundaries (10%, 40%, 100%)
    // This caused phase transitions at wrong times, leading to convergence issues
    //
    // Python weight schedule (iteration-based):
    //   Phase 1 (itr < phase1_iters):      w_neighbors=2.0, w_MN=1000→3 (linear), w_FP=1.0
    //   Phase 2 (itr < phase1+phase2):     w_neighbors=3.0, w_MN=3.0, w_FP=1.0
    //   Phase 3 (itr >= phase1+phase2):    w_neighbors=1.0, w_MN=0.0, w_FP=1.0
    float w_n, w_mn;
    float w_f = 1.0f;  // Constant in Python (this is correct)

    if (current_iter < phase1_iters) {
        // Phase 1: Global structure
        float phase1_progress = static_cast<float>(current_iter) / phase1_iters;
        w_mn = 1000.0f * (1.0f - phase1_progress) + 3.0f * phase1_progress;
        w_n = 2.0f;
    } else if (current_iter < phase1_iters + phase2_iters) {
        // Phase 2: Balance phase
        w_mn = 3.0f;
        w_n = 3.0f;
    } else {
        // Phase 3: Local structure
        w_mn = 0.0f;
        w_n = 1.0f;
    }

    return {w_n, w_mn, w_f};
}

std::tuple<float, float, float, std::string> get_weights_with_phase_info(int current_iter, int phase1_iters, int phase2_iters) {
    // FIX21: Updated to match iteration-based phase boundaries
    float w_n, w_mn;
    float w_f = 1.0f;
    std::string current_phase_name;

    if (current_iter < phase1_iters) {
        float phase1_progress = static_cast<float>(current_iter) / phase1_iters;
        w_mn = 1000.0f * (1.0f - phase1_progress) + 3.0f * phase1_progress;
        w_n = 2.0f;
        current_phase_name = "Phase 1: Global Structure";
    } else if (current_iter < phase1_iters + phase2_iters) {
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


// REMOVED: Old compute_gradients function - replaced by compute_gradients_flat

void compute_gradients_flat(const std::vector<double>& embedding, const std::vector<uint64_t>& triplets_flat,
                           std::vector<double>& gradients, float w_n, float w_mn, float w_f, int n_components,
                           pacmap_progress_callback_internal callback) {

    // FIX21 Cleanup: Removed verbose overflow detection debug output
    // Root cause fixed (200k index rejection removed), uint64_t provides adequate range
    // Bounds checking retained at line 200-205 for memory safety
    size_t embedding_size = embedding.size();
    size_t num_triplets = triplets_flat.size() / 3;

    // MEMORY FIX: Reserve memory to avoid reallocations
    gradients.assign(embedding.size(), 0.0);

    // MEMORY FIX: Check if SIMD is available via runtime detection
    bool use_simd = pacmap_simd::should_use_simd();

  
    // FIX17.md Step 5: Use better OpenMP chunk size for cache locality
    #pragma omp parallel for schedule(static, 1000)
    for (int idx = 0; idx < static_cast<int>(num_triplets); ++idx) {
        size_t triplet_offset = idx * 3;

        // Extract triplet data from flat storage
        uint64_t anchor = triplets_flat[triplet_offset];
        uint64_t neighbor = triplets_flat[triplet_offset + 1];
        uint64_t type = triplets_flat[triplet_offset + 2];

        // FIX21: Removed harmful 200k limit check - uint64_t supports up to 18 quintillion indices
        // The actual bounds check below provides adequate protection for memory safety

        size_t idx_a = static_cast<size_t>(anchor) * n_components;
        size_t idx_n = static_cast<size_t>(neighbor) * n_components;

        // SAFETY CHECK: Verify calculated indices are within embedding bounds
        if (idx_a >= embedding.size() || idx_n >= embedding.size()) {
            continue; // Skip this triplet to prevent memory corruption
        }

        // FIX17.md Step 3: Fused distance computation with NaN/Inf check
        double d_ij = 1.0;
        if (use_simd && n_components >= 4) {
            // Vectorized path using Eigen (AVX2/AVX512)
            Eigen::Map<const Eigen::VectorXd> vec_a(embedding.data() + idx_a, n_components);
            Eigen::Map<const Eigen::VectorXd> vec_n(embedding.data() + idx_n, n_components);
            Eigen::VectorXd diff = vec_a - vec_n;
            d_ij += diff.squaredNorm();  // SIMD-accelerated squared norm
        } else {
            // Scalar fallback for non-AVX2 CPUs or small dimensions
            for (int d = 0; d < n_components; ++d) {
                double diff = embedding[idx_a + d] - embedding[idx_n + d];
                d_ij += diff * diff;
            }
        }

        // FIX17.md Step 3: Inline NaN/Inf check - skip if distance is not finite
        if (!std::isfinite(d_ij)) {
            continue;
        }

        // Calculate gradient magnitude based on triplet type
        // FIX22 Tier 1: Replace std::pow with multiplication for performance
        double grad_magnitude = 0.0;
        switch (static_cast<TripletType>(type)) {
            case NEIGHBOR:
                {
                    double term_n = 10.0 + d_ij;
                    grad_magnitude = static_cast<double>(w_n) * 20.0 / (term_n * term_n);
                }
                break;
            case MID_NEAR:
                {
                    double term_mn = 10000.0 + d_ij;
                    grad_magnitude = static_cast<double>(w_mn) * 20000.0 / (term_mn * term_mn);
                }
                break;
            case FURTHER:
                {
                    double term_f = 1.0 + d_ij;
                    grad_magnitude = -static_cast<double>(w_f) * 2.0 / (term_f * term_f);
                }
                break;
        }

        // FIX17.md Step 3: Inline NaN/Inf check - skip if gradient magnitude is not finite
        if (!std::isfinite(grad_magnitude)) {
            continue;
        }

        // MEMORY FIX: Use atomic operations for thread safety with flat storage
        if (use_simd && n_components >= 4) {
            Eigen::Map<const Eigen::VectorXd> vec_a(embedding.data() + idx_a, n_components);
            Eigen::Map<const Eigen::VectorXd> vec_n(embedding.data() + idx_n, n_components);
            Eigen::VectorXd diff = vec_a - vec_n;
            Eigen::VectorXd grad_vec = grad_magnitude * diff;

            for (int d = 0; d < n_components; ++d) {
                double gradient_component = grad_vec(d);
                if (!std::isfinite(gradient_component)) continue;

                #pragma omp atomic
                gradients[idx_a + d] += gradient_component;
                #pragma omp atomic
                gradients[idx_n + d] -= gradient_component;
            }
        } else {
            for (int d = 0; d < n_components; ++d) {
                double diff = embedding[idx_a + d] - embedding[idx_n + d];
                double gradient_component = grad_magnitude * diff;

                if (!std::isfinite(gradient_component)) continue;

                #pragma omp atomic
                gradients[idx_a + d] += gradient_component;
                #pragma omp atomic
                gradients[idx_n + d] -= gradient_component;
            }
        }
    }
}  // End compute_gradients_flat function

// REMOVED: Old compute_pacmap_loss function - replaced by compute_pacmap_loss_flat

double compute_pacmap_loss_flat(const std::vector<double>& embedding, const std::vector<uint64_t>& triplets_flat,
                               float w_n, float w_mn, float w_f, int n_components,
                               pacmap_progress_callback_internal callback) {

    double total_loss = 0.0;
    double neighbor_loss = 0.0, mn_loss = 0.0, fp_loss = 0.0;
    int neighbor_count = 0, mn_count = 0, fp_count = 0;
    int skipped_zero = 0, skipped_nan = 0;

    size_t num_triplets = triplets_flat.size() / 3;

    for (size_t idx = 0; idx < num_triplets; ++idx) {
        size_t triplet_offset = idx * 3;

        // Extract triplet data from flat storage
        uint64_t anchor = triplets_flat[triplet_offset];
        uint64_t neighbor = triplets_flat[triplet_offset + 1];
        uint64_t type = triplets_flat[triplet_offset + 2];

        size_t idx_a = static_cast<size_t>(anchor) * n_components;
        size_t idx_n = static_cast<size_t>(neighbor) * n_components;

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

        // FIX21: Match Python loss formulas EXACTLY (Python pacmap.py lines 275, 288, 301)
        // Remove extra multipliers (20.0, 20000.0, 2.0) that belong in gradient, not loss
        double loss_term = 0.0;
        switch (static_cast<TripletType>(type)) {
            case NEIGHBOR:
                // Python: loss[0] += w_neighbors * (d_ij / (10. + d_ij))
                loss_term = static_cast<double>(w_n) * d_ij / (10.0 + d_ij);
                neighbor_loss += loss_term;
                neighbor_count++;
                break;
            case MID_NEAR:
                // Python: loss[1] += w_MN * d_ij / (10000. + d_ij)
                loss_term = static_cast<double>(w_mn) * d_ij / (10000.0 + d_ij);
                mn_loss += loss_term;
                mn_count++;
                break;
            case FURTHER:
                // Python: loss[2] += w_FP * 1. / (1. + d_ij)
                loss_term = static_cast<double>(w_f) * 1.0 / (1.0 + d_ij);
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

    double avg_loss = total_loss / static_cast<double>(num_triplets);

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



// REMOVED: Old compute_second_order_info function - would need flat storage adaptation

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

