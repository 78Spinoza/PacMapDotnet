#pragma once

#include "pacmap_model.h"
#include <vector>
#include <random>

// Core triplet sampling functions
extern void sample_triplets(PacMapModel* model, double* data, pacmap_progress_callback_internal callback);
extern void sample_neighbors_pair(PacMapModel* model, const std::vector<double>& normalized_data,
                                 std::vector<Triplet>& neighbor_triplets, pacmap_progress_callback_internal callback = nullptr);
extern void sample_MN_pair(PacMapModel* model, const std::vector<double>& normalized_data,
                         std::vector<Triplet>& mn_triplets, int n_mn, pacmap_progress_callback_internal callback = nullptr);
extern void sample_FP_pair(PacMapModel* model, const std::vector<double>& normalized_data,
                         const std::vector<Triplet>& neighbor_triplets,
                         std::vector<Triplet>& fp_triplets, int n_fp, pacmap_progress_callback_internal callback = nullptr);

// Distance-based sampling helpers
extern void distance_based_sampling(PacMapModel* model, const std::vector<double>& data,
                                   int oversample_factor, int target_pairs, float min_dist, float max_dist,
                                   std::vector<Triplet>& triplets, TripletType type);
extern std::vector<float> compute_distance_percentiles(const std::vector<float>& data, int n_samples, int n_features, PacMapMetric metric);

// HNSW optimization for triplet sampling
extern std::unique_ptr<hnswlib::HierarchicalNSW<float>> create_hnsw_index(
    const double* data, int n_samples, int n_features, PacMapMetric metric,
    int M, int ef_construction, int ef_search, int random_seed, pacmap_progress_callback_internal callback = nullptr);

// Triplet validation and filtering
extern bool is_valid_triplet(const Triplet& triplet, int n_samples);
extern void filter_invalid_triplets(std::vector<Triplet>& triplets, int n_samples);
extern void deduplicate_triplets(std::vector<Triplet>& triplets);

// Sampling strategy utilities
extern void adaptive_sampling_strategy(PacMapModel* model, int n_samples);
extern void balance_triplet_types(std::vector<Triplet>& triplets,
                                  float neighbor_ratio, float mn_ratio, float fp_ratio);

// Distance computation for sampling
extern float compute_sampling_distance(const float* x, const float* y, int n_features, PacMapMetric metric);

// Performance optimization utilities
extern void parallel_triplet_sampling(PacMapModel* model, const std::vector<float>& data,
                                     std::vector<Triplet>& triplets);
extern void batch_distance_computation(const std::vector<float>& data,
                                      std::vector<std::vector<float>>& distance_matrix,
                                      int batch_size, PacMapMetric metric);

// Quality assessment
extern float assess_triplet_quality(const std::vector<Triplet>& triplets,
                                   const std::vector<float>& data, int n_features);
extern void print_sampling_statistics(const std::vector<Triplet>& triplets);

// ¬ PHASE 3: ENHANCED TRIPLET QUALITY VALIDATION
extern void validate_triplet_quality(const std::vector<Triplet>& triplets,
                                    const std::vector<double>& embedding, int n_components,
                                    pacmap_progress_callback_internal callback = nullptr);
extern void analyze_triplet_distance_distributions(const std::vector<Triplet>& triplets,
                                                  const std::vector<double>& embedding, int n_components,
                                                  pacmap_progress_callback_internal callback = nullptr);
extern void check_triplet_coverage(const std::vector<Triplet>& triplets, int n_samples,
                                  pacmap_progress_callback_internal callback = nullptr);
extern void detect_triplet_anomalies(const std::vector<Triplet>& triplets,
                                     const std::vector<double>& embedding, int n_components,
                                     pacmap_progress_callback_internal callback = nullptr);

// Distance computation helper
extern float compute_sampling_distance(const float* x, const float* y, int n_features, PacMapMetric metric);