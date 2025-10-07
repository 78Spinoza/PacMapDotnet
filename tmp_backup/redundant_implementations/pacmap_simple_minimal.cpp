#include "pacmap_simple_wrapper.h"
#include <iostream>

extern "C" {

// Basic error handling
static thread_local char last_error[256] = {0};

// Model creation and destruction - delegate to existing working implementations
PacMapModel* impl_pacmap_create() {
    // Delegate to existing working implementation
    return pacmap_create();
}

void impl_pacmap_destroy(PacMapModel* model) {
    // Delegate to existing working implementation
    pacmap_destroy(model);
}

// REAL PACMAP implementation - calls existing functions, NO DUPLICATES!
int impl_pacmap_fit_with_progress_v2(PacMapModel* model,
    float* data,
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
    float* embedding,
    pacmap_progress_callback_v2 progress_callback,
    int force_exact_knn,
    int M,
    int ef_construction,
    int ef_search,
    int use_quantization,
    int random_seed,
    int autoHNSWParam) {

    std::cout << "=== REAL PACMAP IMPLEMENTATION - USING EXISTING FUNCTIONS ===" << std::endl;
    std::cout << "Parameters: n=" << n_obs << ", d=" << n_dim << ", embed=" << embedding_dim << std::endl;

    if (!model || !data || !embedding) {
        std::cout << "ERROR: Invalid parameters" << std::endl;
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    // The public API will handle model setup internally - no direct access needed

    std::cout << "Model setup complete, calling REAL PACMAP via existing API..." << std::endl;

    // SIMPLE APPROACH: Call the existing working public API function
    // This uses the same code path that C# wrapper uses - NO DUPLICATES!
    std::cout << "Calling pacmap_fit_with_progress_v2 through existing API..." << std::endl;

    int result = pacmap_fit_with_progress_v2(
        model,
        data,
        n_obs,
        n_dim,
        embedding_dim,
        n_neighbors,
        MN_ratio,
        FP_ratio,
        learning_rate,
        n_iters,
        phase1_iters,
        phase2_iters,
        phase3_iters,
        metric,
        embedding,
        progress_callback,
        force_exact_knn,
        M,
        ef_construction,
        ef_search,
        use_quantization,
        random_seed,
        autoHNSWParam
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "ERROR: PACMAP fitting failed with error: " << result << std::endl;
        return result;
    }

    std::cout << "REAL PACMAP algorithm completed through existing API!" << std::endl;

    // The public API handles model fitting internally
    std::cout << "=== REAL PACMAP ALGORITHM COMPLETE ===" << std::endl;

    if (progress_callback) {
        progress_callback("Complete", phase1_iters + phase2_iters + phase3_iters,
                         phase1_iters + phase2_iters + phase3_iters, 100.0f, "PACMAP fitting completed");
    }

    return PACMAP_SUCCESS;
}

// Essential additional functions - delegate to existing working implementations
int impl_pacmap_transform(
    PacMapModel* model,
    float* new_data,
    int n_new_obs,
    int n_dim,
    float* embedding) {
    // Delegate to existing working implementation
    return pacmap_transform(model, new_data, n_new_obs, n_dim, embedding);
}

int impl_pacmap_save_model(PacMapModel* model, const char* filename) {
    // Delegate to existing working implementation
    return pacmap_save_model(model, filename);
}

PacMapModel* impl_pacmap_load_model(const char* filename) {
    // Delegate to existing working implementation
    return pacmap_load_model(filename);
}

const char* impl_pacmap_get_error_message(int error_code) {
    // Delegate to existing working implementation
    return pacmap_get_error_message(error_code);
}

const char* impl_pacmap_get_metric_name(PacMapMetric metric) {
    // Delegate to existing working implementation
    return pacmap_get_metric_name(metric);
}

const char* impl_pacmap_get_version() {
    // Delegate to existing working implementation
    return pacmap_get_version();
}

int impl_pacmap_get_n_components(PacMapModel* model) {
    // Delegate to existing working implementation
    return pacmap_get_n_components(model);
}

int impl_pacmap_get_n_samples(PacMapModel* model) {
    // Delegate to existing working implementation
    return pacmap_get_n_samples(model);
}

int impl_pacmap_is_fitted(PacMapModel* model) {
    // Delegate to existing working implementation
    return pacmap_is_fitted(model);
}

int impl_pacmap_get_n_features(PacMapModel* model) {
    // Use model info API to get n_features
    int n_features = 0, n_neighbors, hnsw_M, hnsw_ef_construction, hnsw_ef_search;
    float mn_ratio, fp_ratio;
    PacMapMetric metric;
    pacmap_get_model_info_simple(model, nullptr, &n_features, nullptr,
                                 &n_neighbors, &mn_ratio, &fp_ratio, &metric,
                                 &hnsw_M, &hnsw_ef_construction, &hnsw_ef_search);
    return n_features;
}

int impl_pacmap_get_n_neighbors(PacMapModel* model) {
    // Use model info API to get n_neighbors
    int n_neighbors = 0, n_features, hnsw_M, hnsw_ef_construction, hnsw_ef_search;
    float mn_ratio, fp_ratio;
    PacMapMetric metric;
    pacmap_get_model_info_simple(model, nullptr, nullptr, nullptr,
                                 &n_neighbors, &mn_ratio, &fp_ratio, &metric,
                                 &hnsw_M, &hnsw_ef_construction, &hnsw_ef_search);
    return n_neighbors;
}

float impl_pacmap_get_mn_ratio(PacMapModel* model) {
    // Use model info API to get mn_ratio
    int n_samples, n_features, n_components, n_neighbors, hnsw_M, hnsw_ef_construction, hnsw_ef_search;
    float mn_ratio = 0.0f, fp_ratio;
    PacMapMetric metric;
    pacmap_get_model_info_simple(model, &n_samples, &n_features, &n_components,
                                 &n_neighbors, &mn_ratio, &fp_ratio, &metric,
                                 &hnsw_M, &hnsw_ef_construction, &hnsw_ef_search);
    return mn_ratio;
}

float impl_pacmap_get_fp_ratio(PacMapModel* model) {
    // Use model info API to get fp_ratio
    int n_samples, n_features, n_components, n_neighbors, hnsw_M, hnsw_ef_construction, hnsw_ef_search;
    float mn_ratio, fp_ratio = 0.0f;
    PacMapMetric metric;
    pacmap_get_model_info_simple(model, &n_samples, &n_features, &n_components,
                                 &n_neighbors, &mn_ratio, &fp_ratio, &metric,
                                 &hnsw_M, &hnsw_ef_construction, &hnsw_ef_search);
    return fp_ratio;
}

PacMapMetric impl_pacmap_get_metric(PacMapModel* model) {
    // Use model info API to get metric
    int n_samples, n_features, n_components, n_neighbors, hnsw_M, hnsw_ef_construction, hnsw_ef_search;
    float mn_ratio, fp_ratio;
    PacMapMetric metric = PACMAP_METRIC_EUCLIDEAN;
    pacmap_get_model_info_simple(model, &n_samples, &n_features, &n_components,
                                 &n_neighbors, &mn_ratio, &fp_ratio, &metric,
                                 &hnsw_M, &hnsw_ef_construction, &hnsw_ef_search);
    return metric;
}

} // extern "C"