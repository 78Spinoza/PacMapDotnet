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
            std::string error_msg = std::string("Failed to create model: ") + e.what();
            send_error_to_callback(error_msg.c_str());
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
        double* data, int n_obs, int n_dim, int embedding_dim,
        int n_neighbors, float MN_ratio, float FP_ratio,
        float learning_rate, int n_iters, int phase1_iters, int phase2_iters, int phase3_iters,
        PacMapMetric metric, double* embedding, pacmap_progress_callback_v2 progress_callback,
        int force_exact_knn, int M, int ef_construction, int ef_search,
        int use_quantization, int random_seed, int autoHNSWParam, float initialization_std_dev) {

        if (!model) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }
        if (!data) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }
        if (!embedding) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        try {
            // Direct call to internal PACMAP implementation
            int result = fit_utils::internal_pacmap_fit_with_progress_v2(model, data, n_obs, n_dim, embedding_dim,
                n_neighbors, MN_ratio, FP_ratio, learning_rate, n_iters, phase1_iters, phase2_iters, phase3_iters,
                metric, embedding, progress_callback, force_exact_knn, M, ef_construction, ef_search,
                use_quantization, random_seed, autoHNSWParam, initialization_std_dev);

            // CRITICAL: Auto-cleanup OpenMP threads after fit completes
            // This prevents segfault during DLL unload by ensuring all threads are terminated
            #ifdef _OPENMP
            omp_set_num_threads(1);
            omp_set_nested(0);
            omp_set_dynamic(0);
            // Force thread pool shutdown
            #pragma omp parallel
            {
                // Single-threaded region forces OpenMP runtime cleanup
            }
            #endif

            return result;
        } catch (const std::exception& e) {
            std::string error_msg = std::string("PACMAP fitting failed: ") + e.what();
            if (progress_callback) {
                progress_callback("ERROR", 0, 1, 0.0f, error_msg.c_str());
            }
            send_error_to_callback(error_msg.c_str());
            snprintf(last_error, sizeof(last_error), "PACMAP fitting failed: %s", e.what());
            return PACMAP_ERROR_FITTING_FAILED;
        } catch (...) {
            std::string error_msg = "PACMAP fitting failed: Unknown error";
            if (progress_callback) {
                progress_callback("ERROR", 0, 1, 0.0f, error_msg.c_str());
            }
            send_error_to_callback(error_msg.c_str());
            snprintf(last_error, sizeof(last_error), "PACMAP fitting failed: Unknown error");
            return PACMAP_ERROR_FITTING_FAILED;
        }
    }

    PACMAP_API int pacmap_transform(PacMapModel* model,
        double* new_data,
        int n_new_obs,
        int n_dim,
        double* embedding) {
        // Direct call to internal transform implementation
        return transform_utils::internal_pacmap_transform(model, new_data, n_new_obs, n_dim, embedding);
    }

    PACMAP_API int pacmap_transform_detailed(PacMapModel* model,
        double* new_data,
        int n_new_obs,
        int n_dim,
        double* embedding,
        int* nn_indices,
        double* nn_distances,
        double* confidence_score,
        int* outlier_level,
        double* percentile_rank,
        double* z_score) {
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
        double* min_embedding_distance,
        double* p95_embedding_distance,
        double* p99_embedding_distance,
        double* mild_embedding_outlier_threshold,
        double* extreme_embedding_outlier_threshold,
        double* mean_embedding_distance,
        double* std_embedding_distance,
        uint32_t* original_space_crc,
        uint32_t* embedding_space_crc,
        uint32_t* model_version_crc,
        float* initialization_std_dev,
        int* always_save_embedding_data,
        double* p25_distance,
        double* p75_distance,
        float* adam_eps) {
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

        // Additional persistence fields
        if (initialization_std_dev) *initialization_std_dev = model->initialization_std_dev;
        if (always_save_embedding_data) *always_save_embedding_data = model->always_save_embedding_data ? 1 : 0;
        if (p25_distance) *p25_distance = model->p25_distance;
        if (p75_distance) *p75_distance = model->p75_distance;
        if (adam_eps) *adam_eps = model->adam_eps;

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

    PACMAP_API int pacmap_use_simple_sgd(PacMapModel* model) {
        // ERROR13 FIX: Switch to simple SGD to match Python reference exactly
        // Sets adam_beta1=0 to disable Adam and use simple SGD: embedding -= lr * gradients
        if (!model) return PACMAP_ERROR_INVALID_PARAMS;
        model->adam_beta1 = 0.0f;  // Disables Adam, enables simple SGD
        model->adam_beta2 = 0.0f;  // Not used in SGD mode
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

    PACMAP_API int pacmap_test_minimal_fit(PacMapModel* model) {
        if (!model) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        // Try to set some basic parameters
        model->n_samples = 10;
        model->n_features = 5;
        model->n_components = 2;

        return PACMAP_SUCCESS;
    }

    // OpenMP cleanup function to prevent segfault on DLL unload
    PACMAP_API void pacmap_cleanup() {
        // Clean up OpenMP threads to prevent segfault on DLL unload
        #ifdef _OPENMP
        // CRITICAL: Force immediate shutdown of ALL OpenMP activity
        // This prevents any lingering threads from causing segfault

        // Step 1: Disable all parallelism immediately
        omp_set_num_threads(1);
        omp_set_nested(0);
        omp_set_dynamic(0);

        // Step 2: Execute a dummy parallel region to force thread pool shutdown
        // This ensures all worker threads are terminated before DLL unload
        #pragma omp parallel
        {
            // This single-threaded region forces OpenMP runtime to clean up
            // the thread pool and terminate worker threads
        }
        #endif
    }

} // extern "C"

// DLL process detach handler for clean OpenMP shutdown
#ifdef _WIN32
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_DETACH:
        // Clean up OpenMP threads before DLL unload to prevent segfault
        #ifdef _OPENMP
        // Force complete OpenMP shutdown
        omp_set_num_threads(1);
        // Reset thread pool completely
        omp_set_nested(0);
        // Additional safety: disable dynamic thread adjustment
        omp_set_dynamic(0);
        #endif
        break;
    }
    return TRUE;
}
#endif