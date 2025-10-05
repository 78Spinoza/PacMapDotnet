#pragma once

#include <vector>
#include <memory>

#ifdef __cplusplus
extern "C" {
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

// Distance metrics
typedef enum {
    PACMAP_METRIC_EUCLIDEAN = 0,
    PACMAP_METRIC_COSINE = 1,
    PACMAP_METRIC_MANHATTAN = 2,
    PACMAP_METRIC_CORRELATION = 3,
    PACMAP_METRIC_HAMMING = 4
} PacMapMetric;

// Progress callback types
typedef void (*pacmap_progress_callback_v2)(
    const char* phase,
    int current,
    int total,
    float percent,
    const char* message
);

// Core PACMAP Model Structure with HNSW support
struct PacMapModel {
    // Basic model parameters
    int n_samples;
    int n_features;
    int n_components;
    int n_neighbors;
    float MN_ratio;
    float FP_ratio;
    float learning_rate;
    int n_iters;
    int phase1_iters;
    int phase2_iters;
    int phase3_iters;
    PacMapMetric metric;

    // Training embedding storage (always needed)
    std::vector<float> training_embedding;

    // HNSW parameters
    int force_exact_knn;
    int M;
    int ef_construction;
    int ef_search;
    int use_quantization;

    // 🎯 CONDITIONAL STORAGE: HNSW indices OR raw training data
    std::vector<float> training_data;  // Only when force_exact_knn = 1
    bool uses_hnsw;                    // true = save HNSW indices, false = save raw data

    // Random seed for reproducibility
    int random_seed;
    bool has_saved_seed;

    // Model state
    bool is_fitted;

    // CRC validation
    uint32_t model_crc;
    uint32_t data_crc;
    uint32_t embedding_crc;

    // Constructor
    PacMapModel() :
        n_samples(0), n_features(0), n_components(2), n_neighbors(10),
        MN_ratio(2.0f), FP_ratio(1.0f), learning_rate(1.0f),
        n_iters(100), phase1_iters(100), phase2_iters(100), phase3_iters(100),
        metric(PACMAP_METRIC_EUCLIDEAN),
        force_exact_knn(0), M(-1), ef_construction(-1), ef_search(-1), use_quantization(0),
        random_seed(-1), has_saved_seed(false), is_fitted(false), uses_hnsw(false),
        model_crc(0), data_crc(0), embedding_crc(0) {}
};

// Core functions
PacMapModel* pacmap_create();
void pacmap_destroy(PacMapModel* model);

int pacmap_fit_with_progress_v2(PacMapModel* model,
    float* data, int n_obs, int n_dim, int embedding_dim,
    int n_neighbors, float MN_ratio, float FP_ratio,
    float learning_rate, int n_iters, int phase1_iters, int phase2_iters, int phase3_iters,
    PacMapMetric metric, float* embedding, pacmap_progress_callback_v2 progress_callback,
    int force_exact_knn, int M, int ef_construction, int ef_search,
    int use_quantization, int random_seed, int autoHNSWParam);

int pacmap_transform(PacMapModel* model,
    float* new_data, int n_new_obs, int n_dim, float* embedding);

int pacmap_save_model(PacMapModel* model, const char* filename);
PacMapModel* pacmap_load_model(const char* filename);

int pacmap_get_model_info_simple(PacMapModel* model,
    int* n_samples, int* n_features, int* n_components, int* n_neighbors,
    float* MN_ratio, float* FP_ratio, PacMapMetric* metric,
    int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search);

int pacmap_get_n_components(PacMapModel* model);
int pacmap_get_n_samples(PacMapModel* model);
int pacmap_get_n_features(PacMapModel* model);
int pacmap_get_n_neighbors(PacMapModel* model);
float pacmap_get_mn_ratio(PacMapModel* model);
float pacmap_get_fp_ratio(PacMapModel* model);
PacMapMetric pacmap_get_metric(PacMapModel* model);
int pacmap_is_fitted(PacMapModel* model);

const char* pacmap_get_error_message(int error_code);
const char* pacmap_get_metric_name(PacMapMetric metric);
const char* pacmap_get_version();

#ifdef __cplusplus
}
#endif