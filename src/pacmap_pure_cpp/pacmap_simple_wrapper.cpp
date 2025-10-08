#include "pacmap_simple_wrapper.h"
#include "pacmap_fit.h"
#include "pacmap_transform.h"
#include "pacmap_persistence.h"
#include "pacmap_distance.h"
#include "pacmap_utils.h"
#include <cstdlib>
#include <iostream>

extern "C" {

    // Simple static error storage
    static thread_local char last_error[256] = {0};

    PACMAP_API PacMapModel* pacmap_create() {
        try {
            return new PacMapModel();
        } catch (const std::exception& e) {
            snprintf(last_error, sizeof(last_error), "Failed to create model: %s", e.what());
            return nullptr;
        }
    }

    PACMAP_API void pacmap_destroy(PacMapModel* model) {
        if (model) {
            delete model;
        }
    }

    
    
    
    PACMAP_API int pacmap_fit_with_progress_v2(PacMapModel* model,
        float* data, int n_obs, int n_dim, int embedding_dim,
        int n_neighbors, float MN_ratio, float FP_ratio,
        float learning_rate, int n_iters, int phase1_iters, int phase2_iters, int phase3_iters,
        PacMapMetric metric, float* embedding, pacmap_progress_callback_v2 progress_callback,
        int force_exact_knn, int M, int ef_construction, int ef_search,
        int use_quantization, int random_seed, int autoHNSWParam, float initialization_std_dev) {
        // Direct call to internal PACMAP implementation
        return fit_utils::internal_pacmap_fit_with_progress_v2(model, data, n_obs, n_dim, embedding_dim,
            n_neighbors, MN_ratio, FP_ratio, learning_rate, n_iters, phase1_iters, phase2_iters, phase3_iters,
            metric, embedding, progress_callback, force_exact_knn, M, ef_construction, ef_search,
            use_quantization, random_seed, autoHNSWParam, initialization_std_dev);
    }

    PACMAP_API int pacmap_transform(PacMapModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding) {
        // Direct call to internal transform implementation
        return transform_utils::internal_pacmap_transform(model, new_data, n_new_obs, n_dim, embedding);
    }

    PACMAP_API int pacmap_transform_detailed(PacMapModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding,
        int* nn_indices,
        float* nn_distances,
        float* confidence_score,
        int* outlier_level,
        float* percentile_rank,
        float* z_score) {
        // Direct call to internal detailed transform implementation
        return transform_utils::internal_pacmap_transform_detailed(model, new_data, n_new_obs, n_dim, embedding,
            nn_indices, nn_distances, confidence_score, outlier_level, percentile_rank, z_score);
    }

    
    // Simplified version for C# interface (matches C# WindowsGetModelInfo signature)
    // Enhanced to include all persistence fields for complete model information
    PACMAP_API int pacmap_get_model_info_simple(PacMapModel* model,
        int* n_samples,
        int* n_features,
        int* n_components,
        int* n_neighbors,
        float* MN_ratio,
        float* FP_ratio,
        PacMapMetric* metric,
        int* hnsw_M,
        int* hnsw_ef_construction,
        int* hnsw_ef_search,
        int* force_exact_knn,
        int* random_seed,
        float* min_embedding_distance,
        float* p95_embedding_distance,
        float* p99_embedding_distance,
        float* mild_embedding_outlier_threshold,
        float* extreme_embedding_outlier_threshold,
        float* mean_embedding_distance,
        float* std_embedding_distance,
        uint32_t* original_space_crc,
        uint32_t* embedding_space_crc,
        uint32_t* model_version_crc) {
        // Direct access to model fields including all persistence fields
        if (!model) return PACMAP_ERROR_INVALID_PARAMS;

        if (n_samples) *n_samples = model->n_samples;
        if (n_features) *n_features = model->n_features;
        if (n_components) *n_components = model->n_components;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (MN_ratio) *MN_ratio = model->mn_ratio;
        if (FP_ratio) *FP_ratio = model->fp_ratio;
        if (metric) *metric = model->metric;
        if (hnsw_M) *hnsw_M = model->hnsw_m;
        if (hnsw_ef_construction) *hnsw_ef_construction = model->hnsw_ef_construction;
        if (hnsw_ef_search) *hnsw_ef_search = model->hnsw_ef_search;

        // New persistence fields
        if (force_exact_knn) *force_exact_knn = model->force_exact_knn ? 1 : 0;
        if (random_seed) *random_seed = model->random_seed;
        if (min_embedding_distance) *min_embedding_distance = model->min_embedding_distance;
        if (p95_embedding_distance) *p95_embedding_distance = model->p95_embedding_distance;
        if (p99_embedding_distance) *p99_embedding_distance = model->p99_embedding_distance;
        if (mild_embedding_outlier_threshold) *mild_embedding_outlier_threshold = model->mild_embedding_outlier_threshold;
        if (extreme_embedding_outlier_threshold) *extreme_embedding_outlier_threshold = model->extreme_embedding_outlier_threshold;
        if (mean_embedding_distance) *mean_embedding_distance = model->mean_embedding_distance;
        if (std_embedding_distance) *std_embedding_distance = model->std_embedding_distance;
        if (original_space_crc) *original_space_crc = model->original_space_crc;
        if (embedding_space_crc) *embedding_space_crc = model->embedding_space_crc;
        if (model_version_crc) *model_version_crc = model->model_version_crc;

        return PACMAP_SUCCESS;
    }

    
  PACMAP_API const char* pacmap_get_error_message(int error_code) {
        // Call the internal implementation
        return internal_pacmap_get_error_message(error_code);
    }

    PACMAP_API const char* pacmap_get_metric_name(PacMapMetric metric) {
        switch (metric) {
            case PACMAP_METRIC_EUCLIDEAN: return "euclidean";
            case PACMAP_METRIC_COSINE: return "cosine";
            case PACMAP_METRIC_MANHATTAN: return "manhattan";
            case PACMAP_METRIC_CORRELATION: return "correlation";
            case PACMAP_METRIC_HAMMING: return "hamming";
            default: return "unknown";
        }
    }

    PACMAP_API int pacmap_get_n_components(PacMapModel* model) {
        return model ? model->n_components : 0;
    }

    PACMAP_API int pacmap_get_n_samples(PacMapModel* model) {
        return model ? model->n_samples : 0;
    }

    PACMAP_API int pacmap_is_fitted(PacMapModel* model) {
        return model && model->is_fitted ? 1 : 0;
    }

    PACMAP_API const char* pacmap_get_version() {
        return PACMAP_WRAPPER_VERSION_STRING;
    }

    PACMAP_API void pacmap_set_always_save_embedding_data(PacMapModel* model, bool always_save) {
        if (model) {
            model->always_save_embedding_data = always_save;
        }
    }

    PACMAP_API bool pacmap_get_always_save_embedding_data(PacMapModel* model) {
        return model ? model->always_save_embedding_data : false;
    }

    PACMAP_API float pacmap_get_learning_rate(PacMapModel* model) {
        return model ? model->learning_rate : 1.0f;
    }

    PACMAP_API float pacmap_get_adam_beta1(PacMapModel* model) {
        return model ? model->adam_beta1 : 0.9f;
    }

    PACMAP_API float pacmap_get_adam_beta2(PacMapModel* model) {
        return model ? model->adam_beta2 : 0.999f;
    }

    PACMAP_API int pacmap_set_learning_rate(PacMapModel* model, float learning_rate) {
        if (!model) return PACMAP_ERROR_INVALID_PARAMS;
        model->learning_rate = learning_rate;
        return PACMAP_SUCCESS;
    }

    PACMAP_API int pacmap_set_adam_beta1(PacMapModel* model, float beta1) {
        if (!model) return PACMAP_ERROR_INVALID_PARAMS;
        model->adam_beta1 = beta1;
        return PACMAP_SUCCESS;
    }

    PACMAP_API int pacmap_set_adam_beta2(PacMapModel* model, float beta2) {
        if (!model) return PACMAP_ERROR_INVALID_PARAMS;
        model->adam_beta2 = beta2;
        return PACMAP_SUCCESS;
    }

    PACMAP_API int pacmap_save_model(PacMapModel* model, const char* filename) {
        return persistence_utils::save_model(model, filename);
    }

    PACMAP_API PacMapModel* pacmap_load_model(const char* filename) {
        return persistence_utils::load_model(filename);
    }

    PACMAP_API void pacmap_set_global_callback(pacmap_progress_callback_v2 callback) {
        // No-op in minimal implementation
    }

    PACMAP_API void pacmap_clear_global_callback() {
        // No-op in minimal implementation
    }

    
} // extern "C"