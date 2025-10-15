#ifndef PACMAP_SIMPLE_WRAPPER_H
#define PACMAP_SIMPLE_WRAPPER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

    // Export macros
#ifdef _WIN32
#ifdef PACMAP_EXPORTS
#define PACMAP_API __declspec(dllexport)
#else
#define PACMAP_API __declspec(dllimport)
#endif
#else
#define PACMAP_API __attribute__((visibility("default")))
#endif

// Error codes
#define PACMAP_SUCCESS 0
#define PACMAP_ERROR_INVALID_PARAMS -1
#define PACMAP_ERROR_MEMORY -2
#define PACMAP_ERROR_NOT_IMPLEMENTED -3
#define PACMAP_ERROR_FILE_IO -4
#define PACMAP_ERROR_MODEL_NOT_FITTED -5
#define PACMAP_ERROR_INVALID_MODEL_FILE -6
#define PACMAP_ERROR_CRC_MISMATCH -7
#define PACMAP_ERROR_QUANTIZATION_FAILURE -8
#define PACMAP_ERROR_OPTIMIZATION_FAILURE -9
#define PACMAP_ERROR_FITTING_FAILED -10

// Version information
#define PACMAP_WRAPPER_VERSION_STRING "2.8.23"

// Distance metrics
    typedef enum {
        PACMAP_METRIC_EUCLIDEAN = 0,
        PACMAP_METRIC_COSINE = 1,
        PACMAP_METRIC_MANHATTAN = 2,
        PACMAP_METRIC_CORRELATION = 3,
        PACMAP_METRIC_HAMMING = 4
    } PacMapMetric;

    // Outlier level enumeration for enhanced safety detection
    typedef enum {
        PACMAP_OUTLIER_NORMAL = 0,      // Within normal range ( p95)
        PACMAP_OUTLIER_UNUSUAL = 1,     // Unusual but acceptable (p95-p99)
        PACMAP_OUTLIER_MILD = 2,        // Mild outlier (p99 to 2.5)
        PACMAP_OUTLIER_EXTREME = 3,     // Extreme outlier (2.5 to 4)
        PACMAP_OUTLIER_NOMANSLAND = 4   // No man's land (> 4)
    } PacMapOutlierLevel;

    // Forward declaration
    typedef struct PacMapModel PacMapModel;

    // Progress callback function types
    typedef void (*pacmap_progress_callback)(int epoch, int total_epochs, float percent);

    // Enhanced progress callback with phase information, time estimates, and warnings
    typedef void (*pacmap_progress_callback_v2)(
        const char* phase,        // Current phase: "Normalizing", "Building HNSW", "Triplet Sampling", etc.
        int current,              // Current progress counter
        int total,                // Total items to process
        float percent,            // Progress percentage (0-100)
        const char* message       // Time estimates, warnings, or NULL for no message
    );

    // Thread-safe callback with user data pointer
    typedef void (*pacmap_progress_callback_v3)(
        const char* phase,        // Current phase: "Normalizing", "Building HNSW", "Triplet Sampling", etc.
        int current,              // Current progress counter
        int total,                // Total items to process
        float percent,            // Progress percentage (0-100)
        const char* message,      // Time estimates, warnings, or NULL for no message
        void* user_data           // User-defined context pointer for thread safety
    );

    // Core functions
    PACMAP_API PacMapModel* pacmap_create();
    PACMAP_API void pacmap_destroy(PacMapModel* model);

    
    
    
    PACMAP_API int pacmap_fit_with_progress_v2(PacMapModel* model,
        double* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float MN_ratio,
        float FP_ratio,
        float learning_rate,
        int n_iters,
        int phase1_iters,
        int phase2_iters,
        int phase3_iters,
        PacMapMetric metric,
        double* embedding,
        pacmap_progress_callback_v2 progress_callback,
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search,
        int use_quantization,
        int random_seed,
        int autoHNSWParam,
        float initialization_std_dev);

  
    // Test function to isolate segfault issue
    PACMAP_API int pacmap_test_minimal_fit(PacMapModel* model);

    // Global callback management functions
    PACMAP_API void pacmap_set_global_callback(pacmap_progress_callback_v2 callback);
    PACMAP_API void pacmap_clear_global_callback();

    // Transform functions
    PACMAP_API int pacmap_transform(PacMapModel* model,
        double* new_data,
        int n_new_obs,
        int n_dim,
        double* embedding);

    // Enhanced transform function with comprehensive safety analysis
    // Returns detailed information about nearest neighbors, confidence, and outlier detection
    // Parameters:
    //   - embedding: Output embedding coordinates [n_new_obs * embedding_dim]
    //   - nn_indices: Output nearest neighbor indices [n_new_obs * n_neighbors] (can be NULL)
    //   - nn_distances: Output nearest neighbor distances [n_new_obs * n_neighbors] (can be NULL)
    //   - confidence_score: Output confidence scores [n_new_obs] (0.0-1.0, can be NULL)
    //   - outlier_level: Output outlier levels [n_new_obs] (PacMapOutlierLevel enum, can be NULL)
    //   - percentile_rank: Output percentile ranks [n_new_obs] (0-100, can be NULL)
    //   - z_score: Output z-scores [n_new_obs] (standard deviations from mean, can be NULL)
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
        double* z_score);

    // Model persistence
    PACMAP_API int pacmap_save_model(PacMapModel* model, const char* filename);
    PACMAP_API PacMapModel* pacmap_load_model(const char* filename);

    // Enhanced model information APIs - PACMAP-specific
    PACMAP_API int pacmap_get_model_info_simple(PacMapModel* model,
        int* n_samples, int* n_features, int* n_components,
        int* n_neighbors, float* mn_ratio, float* fp_ratio,
        PacMapMetric* metric, int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search,
        int* force_exact_knn, int* random_seed,
        double* min_embedding_distance, double* p95_embedding_distance, double* p99_embedding_distance,
        double* mild_embedding_outlier_threshold, double* extreme_embedding_outlier_threshold,
        double* mean_embedding_distance, double* std_embedding_distance,
        uint32_t* original_space_crc, uint32_t* embedding_space_crc, uint32_t* model_version_crc,
        float* initialization_std_dev, int* always_save_embedding_data,
        double* p25_distance, double* p75_distance, float* adam_eps);

    
    PACMAP_API int pacmap_get_triplet_info(PacMapModel* model,
        int* total_triplets, int* neighbor_triplets, int* mid_near_triplets, int* far_triplets);

    PACMAP_API int pacmap_get_optimization_info(PacMapModel* model,
        int* phase1_iters, int* phase2_iters, int* phase3_iters,
        float* learning_rate, int* is_fitted);

    // Utility functions
    PACMAP_API const char* pacmap_get_error_message(int error_code);
    PACMAP_API const char* pacmap_get_metric_name(PacMapMetric metric);
    PACMAP_API int pacmap_get_n_components(PacMapModel* model);
    PACMAP_API int pacmap_get_n_samples(PacMapModel* model);
    PACMAP_API int pacmap_get_n_features(PacMapModel* model);
    PACMAP_API float pacmap_get_mn_ratio(PacMapModel* model);
    PACMAP_API float pacmap_get_fp_ratio(PacMapModel* model);
    PACMAP_API PacMapMetric pacmap_get_metric(PacMapModel* model);
    PACMAP_API int pacmap_get_random_seed(PacMapModel* model);
    PACMAP_API float pacmap_get_learning_rate(PacMapModel* model);
    PACMAP_API float pacmap_get_adam_beta1(PacMapModel* model);
    PACMAP_API float pacmap_get_adam_beta2(PacMapModel* model);
    PACMAP_API int pacmap_get_phase_iters(PacMapModel* model, int* phase1_iters, int* phase2_iters, int* phase3_iters);
    PACMAP_API int pacmap_set_random_seed(PacMapModel* model, int seed);
    PACMAP_API int pacmap_set_learning_rate(PacMapModel* model, float learning_rate);
    PACMAP_API int pacmap_set_adam_beta1(PacMapModel* model, float beta1);
    PACMAP_API int pacmap_set_adam_beta2(PacMapModel* model, float beta2);
    PACMAP_API int pacmap_use_simple_sgd(PacMapModel* model);
    PACMAP_API int pacmap_reset_model(PacMapModel* model);
    PACMAP_API int pacmap_copy_model(PacMapModel* source, PacMapModel** destination);
    PACMAP_API int pacmap_is_fitted(PacMapModel* model);
    PACMAP_API const char* pacmap_get_version();

    // Embedding data preservation options
    PACMAP_API void pacmap_set_always_save_embedding_data(PacMapModel* model, bool always_save);
    PACMAP_API bool pacmap_get_always_save_embedding_data(PacMapModel* model);

    
#ifdef __cplusplus

// Forward declarations for implementation namespaces
namespace transform_utils {
    int pacmap_transform(PacMapModel* model, float* new_data, int n_new_obs, int n_dim, float* embedding);
    int pacmap_transform_detailed(PacMapModel* model, float* new_data, int n_new_obs, int n_dim, float* embedding,
        int* nn_indices, float* nn_distances, float* confidence_score, int* outlier_level, float* percentile_rank, float* z_score);
}

}
#endif

#endif // PACMAP_SIMPLE_WRAPPER_H