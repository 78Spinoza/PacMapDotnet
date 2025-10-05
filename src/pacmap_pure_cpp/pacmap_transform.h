#pragma once

#include "pacmap_model.h"
#include <vector>

// Main transformation functions
extern int pacmap_fit_transform(PacMapModel* model, const float* X, float* embedding_out,
                                uwot_progress_callback_v2 callback);
extern int pacmap_transform(PacMapModel* model, const float* X, float* embedding_out,
                            uwot_progress_callback_v2 callback);

// Transform validation
extern bool validate_transform_input(const PacMapModel* model, const float* X);
extern bool validate_transform_output(const PacMapModel* model, const float* embedding);

// Batch processing for large datasets
extern int batch_transform(PacMapModel* model, const float* X, float* embedding_out,
                          int batch_size, uwot_progress_callback_v2 callback);

// Transform utilities
extern void prepare_new_data(PacMapModel* model, const float* X, std::vector<float>& processed_data);
extern void apply_transform_preprocessing(const std::vector<float>& source_data,
                                        std::vector<float>& target_data,
                                        const PacMapModel* model);

// Transform for new/unseen data points
extern int transform_new_points(PacMapModel* model, const float* new_X, float* embedding_out,
                               int n_new_points, uwot_progress_callback_v2 callback);

// Distance computation in embedding space
extern void compute_embedding_distances(const float* embedding, int n_samples, int n_components,
                                      std::vector<std::vector<float>>& distance_matrix);

// Transform quality assessment
extern float assess_transform_quality(const float* original_embedding, const float* new_embedding,
                                     int n_samples, int n_components);
extern bool detect_transform_anomalies(const float* embedding, int n_samples, int n_components);

// Transform optimization
extern void optimize_transform_parameters(PacMapModel* model, const float* X);
extern void calibrate_transform_for_dataset(PacMapModel* model, int n_samples, int n_features);

// Transform state management
extern void save_transform_state(const PacMapModel* model, const std::string& filename);
extern void load_transform_state(PacMapModel* model, const std::string& filename);

// Transform diagnostics
struct TransformDiagnostics {
    float preprocessing_time_ms = 0.0f;
    float transform_time_ms = 0.0f;
    float quality_score = 0.0f;
    int anomaly_count = 0;
    bool is_valid = true;
};

extern TransformDiagnostics run_transform_with_diagnostics(PacMapModel* model, const float* X,
                                                          float* embedding_out,
                                                          uwot_progress_callback_v2 callback);

// Advanced transform features
extern void incremental_transform_update(PacMapModel* model, const float* new_X,
                                        float* embedding_out, int n_new_points);
extern void adaptive_transform_resolution(PacMapModel* model, const float* X, float* embedding_out);

// Transform validation utilities
extern bool check_embedding_integrity(const float* embedding, int n_samples, int n_components);
extern void validate_embedding_statistics(const float* embedding, int n_samples, int n_components);