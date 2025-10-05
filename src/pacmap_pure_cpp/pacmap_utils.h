#pragma once

#include "pacmap_model.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

// Parameter validation functions
extern int validate_parameters(PacMapModel* model);
extern void validate_metric_data(const float* data, int n_obs, int n_dim, PacMapMetric metric);
extern bool check_memory_requirements(int n_samples, int n_features, int n_neighbors);
extern void auto_tune_parameters(PacMapModel* model, int n_samples);

// Edge case detection
extern bool detect_degenerate_cases(int n_samples, int n_features);
extern bool check_for_nan_inf(const float* data, int size);
extern bool validate_triplet_distribution(const std::vector<Triplet>& triplets, int n_samples);

// Performance monitoring
extern void start_performance_timer(PacMapModel* model);
extern void record_performance_stats(PacMapModel* model, const std::string& operation);
extern PerformanceStats get_performance_stats(const PacMapModel* model);

// Distance computation utilities
extern float compute_distance(const float* x, const float* y, int n_features, PacMapMetric metric);

// Data normalization and preprocessing
extern void normalize_data(std::vector<float>& data, int n_samples, int n_features, PacMapMetric metric);
extern void standardize_data(std::vector<float>& data, int n_samples, int n_features);

// Random number generation utilities
extern std::mt19937 get_seeded_rng(int seed);
extern void set_random_seed(PacMapModel* model, int seed);

// Memory management utilities
extern void* aligned_malloc(size_t size, size_t alignment);
extern void aligned_free(void* ptr);

// String utilities
extern std::string format_bytes(size_t bytes);
extern std::string format_duration(double milliseconds);

// Validation utilities
extern bool is_valid_parameter_combination(float mn_ratio, float fp_ratio, int n_neighbors);
extern bool is_supported_metric(PacMapMetric metric);
extern bool is_valid_embedding_dimension(int n_components);

// Debug utilities
extern void print_model_info(const PacMapModel* model);
extern void print_triplet_stats(const std::vector<Triplet>& triplets);
extern void print_performance_stats(const PerformanceStats& stats);