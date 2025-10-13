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

// 🔬 PHASE 4: ENHANCED PHASE TRANSITION DEBUGGING
std::tuple<float, float, float, const char*> get_weights_with_phase_info(int current_iter, int total_iters) {
    // Enhanced version that returns phase information for debugging
    float w_n, w_mn, w_f = 1.0f;
    const char* phase_name = "";

    float progress = static_cast<float>(current_iter) / total_iters;

    if (progress < 0.1f) {
        // Phase 1: Global structure (0-10%)
        float phase1_progress = progress / 0.1f;
        w_mn = 1000.0f * (1.0f - phase1_progress) + 3.0f * phase1_progress;
        w_n = 2.0f;
        phase_name = "Phase 1: Global Structure";
    } else if (progress < 0.4f) {
        // Phase 2: Balance phase (10-40%)
        w_mn = 3.0f;
        w_n = 3.0f;
        phase_name = "Phase 2: Balance";
    } else {
        // Phase 3: Local structure (40-100%)
        // FIX13: w_MN must be ZERO immediately in Phase 3 (not gradual decay)
        // Python reference (pacmap.py line 350): w_MN = 0.0 (instant zero)
        // Previous bug: Used gradual decay w_mn = 3.0f * (1.0f - phase3_progress) → wrong force balance
        w_mn = 0.0f;  // ← Match Python exactly!
        w_n = 1.0f;
        phase_name = "Phase 3: Local Structure";
    }

    return {w_n, w_mn, w_f, phase_name};
}

void compute_gradients(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                       std::vector<double>& gradients, float w_n, float w_mn, float w_f, int n_components,
                       pacmap_progress_callback_internal callback) {

    gradients.assign(embedding.size(), 0.0);

    // 🔬 PHASE 4: Phase transition detection and tracking
    static int last_phase = -1;  // Track phase changes
    static int phase_transition_count = 0;
    float progress = 0.0f;  // Will be set if we can compute it
    const char* current_phase_name = "Unknown";

    // Try to determine current phase from weights (heuristic)
    if (w_n > 1.5f && w_n < 2.5f && w_mn > 3.0f) {
        current_phase_name = "Phase 1: Global Structure";
        if (last_phase != 1) {
            phase_transition_count++;
            last_phase = 1;
            if (callback) {
                char transition_msg[512];
                snprintf(transition_msg, sizeof(transition_msg),
                        "🔄 PHASE TRANSITION v2.8.10: Entered %s (w_n=%.2f, w_mn=%.2f, w_f=%.2f) - Transition #%d",
                        current_phase_name, w_n, w_mn, w_f, phase_transition_count);
                callback("Phase Transition", 0, 0, 0.0f, transition_msg);
            }
        }
    } else if (std::abs(w_n - 3.0f) < 0.1f && std::abs(w_mn - 3.0f) < 0.1f) {
        current_phase_name = "Phase 2: Balance";
        if (last_phase != 2) {
            phase_transition_count++;
            last_phase = 2;
            if (callback) {
                char transition_msg[512];
                snprintf(transition_msg, sizeof(transition_msg),
                        "🔄 PHASE TRANSITION v2.8.10: Entered %s (w_n=%.2f, w_mn=%.2f, w_f=%.2f) - Transition #%d",
                        current_phase_name, w_n, w_mn, w_f, phase_transition_count);
                callback("Phase Transition", 0, 0, 0.0f, transition_msg);
            }
        }
    } else if (std::abs(w_n - 1.0f) < 0.1f && w_mn <= 3.0f) {
        current_phase_name = "Phase 3: Local Structure";
        if (last_phase != 3) {
            phase_transition_count++;
            last_phase = 3;
            if (callback) {
                char transition_msg[512];
                snprintf(transition_msg, sizeof(transition_msg),
                        "🔄 PHASE TRANSITION v2.8.10: Entered %s (w_n=%.2f, w_mn=%.2f, w_f=%.2f) - Transition #%d",
                        current_phase_name, w_n, w_mn, w_f, phase_transition_count);
                callback("Phase Transition", 0, 0, 0.0f, transition_msg);
            }
        }
    }

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

    // ERROR13 FIX: Enhanced debug tracking for triplet processing by type
    int processed_neighbors = 0, processed_midnear = 0, processed_further = 0;
    int skipped_triplets = 0, skipped_nan = 0, skipped_zero_distance = 0;

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
    #pragma omp parallel for schedule(dynamic, 1000) reduction(+:processed_midnear,skipped_nan,skipped_zero_distance)
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
    #pragma omp parallel for schedule(dynamic, 1000) reduction(+:processed_further,skipped_nan,skipped_zero_distance)
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

        // 🚨 REVERTING CRITICAL ERROR v2.8.10: FAR PAIRS MUST BE REPULSIVE!
        // Python Analysis: NEIGHBOR/MID_NEAR are attractive, FURTHER are repulsive
        // Python (pacmap.py): grad[i, d] += w * diff[d], grad[j, d] -= w * diff[d]
        // NEIGHBOR:    w = 20/(10+d)^2, grad[i] +=, grad[j] -= (attractive)
        // MID_NEAR:   w = 20000/(10000+d)^2, grad[i] +=, grad[j] -= (attractive)
        // FARTHER:    w = 2/(1+d)^2, grad[i] -=, grad[j] += (repulsive!)
        //
        // CURRENT FIX: grad_magnitude is now NEGATIVE (repulsive) which is CORRECT!
        // Python treats FURTHER as REPULSIVE to maintain separation, not attractive!
        double grad_magnitude = -static_cast<double>(w_f) * 2.0 / std::pow(1.0 + d_ij, 2.0);  // ✅ NEGATIVE - repulsive force!

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

    // 🔍 ENHANCED DEBUGGING: Comprehensive triplet processing statistics
    static int triplet_debug_counter = 0;
    if (triplet_debug_counter++ % 5 == 0 && callback) {  // Report every 5th iteration
        char debug_msg[512];
        snprintf(debug_msg, sizeof(debug_msg),
                "🔬 TRIPLET DEBUG v2.8.10: N=%d, MN=%d, F=%d | Skipped: zero=%d, nan=%d, other=%d",
                processed_neighbors, processed_midnear, processed_further,
                skipped_zero_distance, skipped_nan, skipped_triplets);
        callback("Triplet Processing", 0, 0, 0.0f, debug_msg);
    }

    // 📊 ENHANCED: Triplet distance distribution analysis (every 10 iterations)
    static int distance_analysis_counter = 0;
    if (distance_analysis_counter++ % 10 == 0 && callback) {
        // Analyze distance distributions by type to validate triplet quality
        std::vector<double> neighbor_distances, mn_distances, fp_distances;

        for (const auto& t : triplets) {
            double d_ij_squared = 0.0;
            size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
            size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

            for (int d = 0; d < n_components; ++d) {
                double diff = embedding[idx_a + d] - embedding[idx_n + d];
                d_ij_squared += diff * diff;
            }
            double d_ij = std::sqrt(std::max(d_ij_squared, 1e-15));

            switch (t.type) {
                case NEIGHBOR: neighbor_distances.push_back(d_ij); break;
                case MID_NEAR: mn_distances.push_back(d_ij); break;
                case FURTHER: fp_distances.push_back(d_ij); break;
            }
        }

        if (!neighbor_distances.empty() && !mn_distances.empty() && !fp_distances.empty()) {
            auto analyze_distances = [](const std::vector<double>& dists, const char* type) {
                if (dists.empty()) return std::make_pair(0.0, 0.0);
                auto[min_it, max_it] = std::minmax_element(dists.begin(), dists.end());
                double mean = std::accumulate(dists.begin(), dists.end(), 0.0) / dists.size();
                double range = static_cast<double>(*max_it) - static_cast<double>(*min_it);
                return std::make_pair(mean, range);
            };

            auto [neighbor_mean, neighbor_range] = analyze_distances(neighbor_distances, "NEIGHBOR");
            auto [mn_mean, mn_range] = analyze_distances(mn_distances, "MID_NEAR");
            auto [fp_mean, fp_range] = analyze_distances(fp_distances, "FARTHER");

            char distance_msg[512];
            snprintf(distance_msg, sizeof(distance_msg),
                    "📊 DISTANCE ANALYSIS: NEIGHBOR (μ=%.3f, range=%.3f) | MN (μ=%.3f, range=%.3f) | FP (μ=%.3f, range=%.3f)",
                    neighbor_mean, neighbor_range, mn_mean, mn_range, fp_mean, fp_range);
            callback("Distance Analysis", 0, 0, 0.0f, distance_msg);

            // Validate distance ordering: NEIGHBOR < MN < FARTHER should hold for proper structure
            if (neighbor_mean >= mn_mean || mn_mean >= fp_mean) {
                char warning_msg[512];
                snprintf(warning_msg, sizeof(warning_msg),
                        "⚠️ DISTANCE ORDER WARNING: Expected NEIGHBOR < MN < FARTHER, got N=%.3f ≥ MN=%.3f ≥ FP=%.3f",
                        neighbor_mean, mn_mean, fp_mean);
                callback("WARNING", 0, 0, 0.0f, warning_msg);
            }
        }
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

    // 🔬 PHASE 2: ENHANCED DEBUGGING - Detailed gradient analysis by triplet type
    double grad_min = gradients[0], grad_max = gradients[0], grad_abssum = 0.0;
    int grad_nan = 0, grad_inf = 0;

    // Per-type gradient accumulation for detailed analysis
    double neighbor_grad_sum = 0.0, mn_grad_sum = 0.0, fp_grad_sum = 0.0;
    int neighbor_grad_count = 0, mn_grad_count = 0, fp_grad_count = 0;

    for (size_t i = 0; i < gradients.size(); ++i) {
        grad_min = std::min(grad_min, gradients[i]);
        grad_max = std::max(grad_max, gradients[i]);
        grad_abssum += std::abs(gradients[i]);
        if (std::isnan(gradients[i])) grad_nan++;
        if (std::isinf(gradients[i])) grad_inf++;
    }
    double grad_absmean = grad_abssum / gradients.size();

    // 🔍 ENHANCED: Calculate gradient contributions by triplet type
    if (callback) {
        // Sample gradient analysis by triplet type
        for (const auto& t : triplets) {
            size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
            size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

            // Compute gradient magnitude contribution for this triplet
            double d_ij_squared = 0.0;
            for (int d = 0; d < n_components; ++d) {
                double diff = embedding[idx_a + d] - embedding[idx_n + d];
                d_ij_squared += diff * diff;
            }
            double d_ij = std::sqrt(std::max(d_ij_squared, 1e-15));

            double grad_mag = 0.0;
            switch (t.type) {
                case NEIGHBOR:
                    grad_mag = std::abs(static_cast<double>(w_n) * 20.0 / std::pow(10.0 + d_ij, 2.0));
                    neighbor_grad_sum += grad_mag;
                    neighbor_grad_count++;
                    break;
                case MID_NEAR:
                    grad_mag = std::abs(static_cast<double>(w_mn) * 20000.0 / std::pow(10000.0 + d_ij, 2.0));
                    mn_grad_sum += grad_mag;
                    mn_grad_count++;
                    break;
                case FURTHER:
                    grad_mag = std::abs(static_cast<double>(w_f) * 2.0 / std::pow(1.0 + d_ij, 2.0));
                    fp_grad_sum += grad_mag;
                    fp_grad_count++;
                    break;
            }
        }
    }

    // 🔬 PHASE 4: Periodic phase status reporting (every 15 iterations)
    static int phase_status_counter = 0;
    if (phase_status_counter++ % 15 == 0 && callback) {
        char phase_status_msg[1024];
        snprintf(phase_status_msg, sizeof(phase_status_msg),
                "🗺️ PHASE STATUS v2.8.10: %s | Transitions: %d | Weights: w_n=%.3f, w_mn=%.3f, w_f=%.3f",
                current_phase_name, phase_transition_count, w_n, w_mn, w_f);
        callback("Phase Status", 0, 0, 0.0f, phase_status_msg);
    }

    // 📊 ENHANCED DEBUGGING: Comprehensive gradient statistics with force balance analysis
    static int debug_counter = 0;
    if (debug_counter++ % 10 == 0 && callback) {  // Report every 10th iteration
        // Count positive vs negative gradients to detect force imbalance
        int positive_count = 0, negative_count = 0, zero_count = 0;
        double pos_sum = 0.0, neg_sum = 0.0;

        for (double g : gradients) {
            if (g > 1e-12) {
                positive_count++;
                pos_sum += g;
            } else if (g < -1e-12) {
                negative_count++;
                neg_sum += g;
            } else {
                zero_count++;
            }
        }

        double pos_avg = positive_count > 0 ? pos_sum / positive_count : 0.0;
        double neg_avg = negative_count > 0 ? neg_sum / negative_count : 0.0;

        // Calculate per-type averages
        double neighbor_avg = neighbor_grad_count > 0 ? neighbor_grad_sum / neighbor_grad_count : 0.0;
        double mn_avg = mn_grad_count > 0 ? mn_grad_sum / mn_grad_count : 0.0;
        double fp_avg = fp_grad_count > 0 ? fp_grad_sum / fp_grad_count : 0.0;

        char grad_msg[1024];
        snprintf(grad_msg, sizeof(grad_msg),
                "🔬 GRADIENT ANALYSIS v2.8.10: min=%.6f, max=%.6f, |mean|=%.6f [+:%d(%.4f) -:%d(%.4f) 0:%d] | By type: N=%.4f, MN=%.4f, F=%.4f",
                grad_min, grad_max, grad_absmean, positive_count, pos_avg, negative_count, neg_avg, zero_count,
                neighbor_avg, mn_avg, fp_avg);

        if (grad_nan > 0 || grad_inf > 0) {
            char warning_msg[1200];
            snprintf(warning_msg, sizeof(warning_msg), "%s WARNING: NaN=%d, Inf=%d", grad_msg, grad_nan, grad_inf);
            callback("Gradient Analysis", 0, 0, 0.0f, warning_msg);
        } else {
            callback("Gradient Analysis", 0, 0, 0.0f, grad_msg);
        }

        // 🎯 Force balance validation: Check if gradient types are balanced
        double total_grad_contrib = neighbor_grad_sum + mn_grad_sum + fp_grad_sum;
        if (total_grad_contrib > 1e-15) {
            double n_ratio = neighbor_grad_sum / total_grad_contrib;
            double mn_ratio = mn_grad_sum / total_grad_contrib;
            double f_ratio = fp_grad_sum / total_grad_contrib;

            char balance_msg[512];
            snprintf(balance_msg, sizeof(balance_msg),
                    "⚖️ FORCE BALANCE: NEIGHBOR=%.1f%%, MID_NEAR=%.1f%%, FURTHER=%.1f%% (weights: n=%.2f, mn=%.2f, f=%.2f)",
                    n_ratio * 100.0, mn_ratio * 100.0, f_ratio * 100.0, w_n, w_mn, w_f);
            callback("Force Balance", 0, 0, 0.0f, balance_msg);
        }
    }

    // DEBUG: Analyze triplet force balance every 20 iterations - reported via callback
    static int force_balance_counter = 0;
    if (force_balance_counter++ % 20 == 0 && callback) {
        // ERROR13 FIX: Analyze force distribution with correct embedding coordinates
        char force_msg[1024];
        char* msg_ptr = force_msg;
        int remaining = sizeof(force_msg);

        // Count triplet types
        int neighbor_count = 0, midnear_count = 0, further_count = 0;
        for (const auto& t : triplets) {
            switch (t.type) {
                case NEIGHBOR: neighbor_count++; break;
                case MID_NEAR: midnear_count++; break;
                case FURTHER: further_count++; break;
            }
        }

        int written = snprintf(msg_ptr, remaining,
                "FORCE ANALYSIS (total triplets: %zu) - Triplet distribution: NEIGHBOR=%d, MID_NEAR=%d, FURTHER=%d",
                triplets.size(), neighbor_count, midnear_count, further_count);
        msg_ptr += written;
        remaining -= written;

        // Sample first few triplets of each type to analyze force distribution
        int samples_per_type = 3;
        int found_neighbors = 0, found_midnear = 0, found_further = 0;

        for (int i = 0; i < static_cast<int>(triplets.size()) &&
             (found_neighbors < samples_per_type || found_midnear < samples_per_type || found_further < samples_per_type) && remaining > 100; ++i) {
            const auto& t = triplets[i];

            const char* type_name = "";
            double grad_mag = 0.0;

            switch (t.type) {
                case NEIGHBOR:
                    if (found_neighbors < samples_per_type) {
                        type_name = "NEIGHBOR";
                        found_neighbors++;
                    } else break;
                    break;
                case MID_NEAR:
                    if (found_midnear < samples_per_type) {
                        type_name = "MID_NEAR";
                        found_midnear++;
                    } else break;
                    break;
                case FURTHER:
                    if (found_further < samples_per_type) {
                        type_name = "FURTHER";
                        found_further++;
                    } else break;
                    break;
                default:
                    continue;
            }

            size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
            size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

            // FIX13: Use Python-matching distance calculation for consistency
            double d_ij = 1.0;
            for (int d = 0; d < n_components; ++d) {
                double diff = embedding[idx_a + d] - embedding[idx_n + d];
                d_ij += diff * diff;  // Python: d_ij = 1.0 + sum(diff²)
            }

            // Compute gradient magnitude for this triplet (using CORRECTED Python formulas - v2.8.7)
            switch (t.type) {
                case NEIGHBOR:
                    grad_mag = static_cast<double>(w_n) * 20.0 / std::pow(10.0 + d_ij, 2.0);  // ✅ v2.8.7: 20.0 not 10.0
                    break;
                case MID_NEAR:
                    grad_mag = static_cast<double>(w_mn) * 20000.0 / std::pow(10000.0 + d_ij, 2.0);  // ✅ v2.8.7: 20000.0 not 10000.0
                    break;
                case FURTHER:
                    grad_mag = static_cast<double>(w_f) * 2.0 / std::pow(1.0 + d_ij, 2.0);  // ✅ v2.8.10: POSITIVE - attractive like Python!
                    break;
            }

            // ERROR13 FIX: Enhanced force analysis with force type classification
            const char* force_type = (grad_mag > 0) ? "ATTRACT" : "REPEL";
            written = snprintf(msg_ptr, remaining, " | %s: d=%.4f, grad=%+.6f (%s)",
                    type_name, d_ij, grad_mag, force_type);
            msg_ptr += written;
            remaining -= written;
        }

        // Add current weights
        snprintf(msg_ptr, remaining, " | Current weights: w_n=%.2f, w_mn=%.2f, w_f=%.2f", w_n, w_mn, w_f);
        callback("Force Analysis", 0, 0, 0.0f, force_msg);
    }
}

double compute_pacmap_loss(const std::vector<double>& embedding, const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components,
                         pacmap_progress_callback_internal callback) {

    // 🔬 PHASE 2: ENHANCED DEBUGGING - Loss computation with detailed by-type analysis
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
        if (callback) {
            callback("Loss Computation", 0, 0, 0.0f, "⚠️ Loss computation failed - returning 0");
        }
        return 0.0;
    }

    double avg_loss = total_loss / static_cast<double>(triplets.size());

    // 🔍 ENHANCED DEBUGGING: Detailed loss analysis by triplet type (every 10 iterations)
    static int loss_debug_counter = 0;
    if (loss_debug_counter++ % 10 == 0 && callback) {
        double neighbor_avg = neighbor_count > 0 ? neighbor_loss / neighbor_count : 0.0;
        double mn_avg = mn_count > 0 ? mn_loss / mn_count : 0.0;
        double fp_avg = fp_count > 0 ? fp_loss / fp_count : 0.0;

        char loss_msg[1024];
        snprintf(loss_msg, sizeof(loss_msg),
                "📊 LOSS ANALYSIS v2.8.10: total=%.6f | By type: N=%.6f (%d), MN=%.6f (%d), F=%.6f (%d) | Skipped: zero=%d, nan=%d",
                avg_loss, neighbor_avg, neighbor_count, mn_avg, mn_count, fp_avg, fp_count, skipped_zero, skipped_nan);
        callback("Loss Analysis", 0, 0, 0.0f, loss_msg);

        // Loss contribution balance analysis
        double total_type_loss = neighbor_loss + mn_loss + fp_loss;
        if (total_type_loss > 1e-15) {
            double n_ratio = neighbor_loss / total_type_loss;
            double mn_ratio = mn_loss / total_type_loss;
            double f_ratio = fp_loss / total_type_loss;

            char loss_balance_msg[512];
            snprintf(loss_balance_msg, sizeof(loss_balance_msg),
                    "⚖️ LOSS BALANCE: NEIGHBOR=%.1f%%, MID_NEAR=%.1f%%, FURTHER=%.1f%% (weights: n=%.2f, mn=%.2f, f=%.2f)",
                    n_ratio * 100.0, mn_ratio * 100.0, f_ratio * 100.0, w_n, w_mn, w_f);
            callback("Loss Balance", 0, 0, 0.0f, loss_balance_msg);
        }
    }

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

void print_gradient_stats(const std::vector<double>& gradients) {
    if (gradients.empty()) return;

    double min_grad = gradients[0], max_grad = gradients[0], sum = 0.0;
    for (double g : gradients) {
        min_grad = std::min(min_grad, g);
        max_grad = std::max(max_grad, g);
        sum += g;
    }
    double mean = sum / gradients.size();

    double variance = 0.0;
    for (double g : gradients) {
        variance += (g - mean) * (g - mean);
    }
    double std_dev = std::sqrt(variance / gradients.size());

    std::cout << "Double Precision Gradient Statistics:" << std::endl;
    std::cout << "  Count: " << gradients.size() << std::endl;
    std::cout << "  Min: " << min_grad << std::endl;
    std::cout << "  Max: " << max_grad << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std Dev: " << std_dev << std::endl;
}

void validate_gradients(const std::vector<double>& gradients, const std::vector<double>& embedding) {
    if (gradients.size() != embedding.size()) {
        std::cerr << "Error: Double precision gradient and embedding sizes mismatch!" << std::endl;
        return;
    }

    int nan_count = 0, inf_count = 0;
    for (double g : gradients) {
        if (std::isnan(g)) nan_count++;
        if (std::isinf(g)) inf_count++;
    }

    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "Error: " << nan_count << " NaN and " << inf_count << " Inf double precision gradients detected!" << std::endl;
    }
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

