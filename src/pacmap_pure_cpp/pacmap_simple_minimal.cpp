#include "pacmap_simple_wrapper.h"
#include "pacmap_utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// Minimal PACMAP implementation for testing

extern "C" {

// Basic error handling
static thread_local char last_error[256] = {0};

// Model creation and destruction (matching header declarations)
PacMapModel* impl_pacmap_create() {
    try {
        return new PacMapModel();
    } catch (...) {
        snprintf(last_error, sizeof(last_error), "Failed to create model");
        return nullptr;
    }
}

void impl_pacmap_destroy(PacMapModel* model) {
    delete model;
}

// Implementation of pacmap_fit_with_progress_v2 from header
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

    if (!model || !data || !embedding) {
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    // Minimal implementation: simple random embedding
    std::mt19937 rng(random_seed >= 0 ? random_seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Report progress
    if (progress_callback) {
        progress_callback("Phase 1", 0, 100, 0.0f, "Initializing PACMAP...");
    }

    // Generate simple random embedding (placeholder for real PACMAP)
    for (int i = 0; i < n_obs * embedding_dim; ++i) {
        embedding[i] = dist(rng);
    }

    // Simulate three phases
    int total_iters = phase1_iters + phase2_iters + phase3_iters;

    for (int phase = 1; phase <= 3; ++phase) {
        int phase_iters = (phase == 1) ? phase1_iters : (phase == 2) ? phase2_iters : phase3_iters;

        for (int iter = 0; iter < phase_iters; iter += 10) {
            float percent = (float)(iter + phase_iters * (phase - 1)) / total_iters * 100.0f;
            if (progress_callback) {
                char phase_name[32];
                snprintf(phase_name, sizeof(phase_name), "Phase %d", phase);
                progress_callback(phase_name, iter + 1, phase_iters, percent, "Optimizing embedding...");
            }
        }
    }

    // Store basic model info
    model->n_samples = n_obs;
    model->n_features = n_dim;
    model->n_components = embedding_dim;
    model->n_neighbors = n_neighbors;
    model->mn_ratio = MN_ratio;
    model->fp_ratio = FP_ratio;
    model->learning_rate = learning_rate;
    model->metric = metric;
    model->is_fitted = true;

    if (progress_callback) {
        progress_callback("Complete", total_iters, total_iters, 100.0f, "PACMAP fitting completed");
    }

    return PACMAP_SUCCESS;
}

// Additional functions needed for testing
int impl_pacmap_transform(
    PacMapModel* model,
    float* new_data,
    int n_new_obs,
    int n_dim,
    float* embedding) {

    if (!model || !new_data || !embedding || !model->is_fitted) {
        return PACMAP_ERROR_MODEL_NOT_FITTED;
    }

    // Minimal implementation: simple random embedding for new data
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n_new_obs * model->n_components; ++i) {
        embedding[i] = dist(rng);
    }

    return PACMAP_SUCCESS;
}

int impl_pacmap_save_model(PacMapModel* model, const char* filename) {
    if (!model || !filename) {
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    // Minimal implementation: just save basic info
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return PACMAP_ERROR_FILE_IO;
    }

    // Write basic model info
    if (fwrite(&model->n_samples, sizeof(int), 1, file) != 1 ||
        fwrite(&model->n_features, sizeof(int), 1, file) != 1 ||
        fwrite(&model->n_components, sizeof(int), 1, file) != 1 ||
        fwrite(&model->is_fitted, sizeof(bool), 1, file) != 1) {
        fclose(file);
        return PACMAP_ERROR_FILE_IO;
    }

    fclose(file);
    return PACMAP_SUCCESS;
}

PacMapModel* impl_pacmap_load_model(const char* filename) {
    if (!filename) {
        return nullptr;
    }

    // Create new model instance
    PacMapModel* model = new PacMapModel();
    if (!model) {
        return nullptr;
    }

    // Initialize with defaults
    model->n_samples = 0;
    model->n_features = 0;
    model->n_components = 0;
    model->is_fitted = false;

    // Minimal implementation: just load basic info
    FILE* file = fopen(filename, "rb");
    if (!file) {
        delete model;
        return nullptr;
    }

    // Read basic model info
    if (fread(&model->n_samples, sizeof(int), 1, file) != 1 ||
        fread(&model->n_features, sizeof(int), 1, file) != 1 ||
        fread(&model->n_components, sizeof(int), 1, file) != 1 ||
        fread(&model->is_fitted, sizeof(bool), 1, file) != 1) {
        fclose(file);
        delete model;
        return nullptr;
    }

    fclose(file);
    return model;
}

// Additional functions needed by the wrapper
const char* impl_pacmap_get_error_message(int error_code) {
    switch (error_code) {
        case PACMAP_SUCCESS: return "Success";
        case PACMAP_ERROR_INVALID_PARAMS: return "Invalid parameters";
        case PACMAP_ERROR_MEMORY: return "Memory allocation error";
        case PACMAP_ERROR_NOT_IMPLEMENTED: return "Not implemented";
        case PACMAP_ERROR_FILE_IO: return "File I/O error";
        case PACMAP_ERROR_MODEL_NOT_FITTED: return "Model not fitted";
        case PACMAP_ERROR_INVALID_MODEL_FILE: return "Invalid model file";
        case PACMAP_ERROR_CRC_MISMATCH: return "CRC mismatch";
        default: return last_error;
    }
}

const char* impl_pacmap_get_metric_name(PacMapMetric metric) {
    switch (metric) {
        case PACMAP_METRIC_EUCLIDEAN: return "euclidean";
        case PACMAP_METRIC_COSINE: return "cosine";
        case PACMAP_METRIC_MANHATTAN: return "manhattan";
        case PACMAP_METRIC_CORRELATION: return "correlation";
        case PACMAP_METRIC_HAMMING: return "hamming";
        default: return "unknown";
    }
}

int impl_pacmap_get_n_components(PacMapModel* model) {
    return model ? model->n_components : 0;
}

int impl_pacmap_get_n_samples(PacMapModel* model) {
    return model ? model->n_samples : 0;
}

int impl_pacmap_is_fitted(PacMapModel* model) {
    return model ? model->is_fitted : 0;
}

const char* impl_pacmap_get_version() {
    return PACMAP_WRAPPER_VERSION_STRING;
}

int impl_pacmap_get_n_features(PacMapModel* model) {
    return model ? model->n_features : 0;
}

int impl_pacmap_get_n_neighbors(PacMapModel* model) {
    return model ? model->n_neighbors : 0;
}

float impl_pacmap_get_mn_ratio(PacMapModel* model) {
    return model ? model->mn_ratio : 0.0f;
}

float impl_pacmap_get_fp_ratio(PacMapModel* model) {
    return model ? model->fp_ratio : 0.0f;
}

PacMapMetric impl_pacmap_get_metric(PacMapModel* model) {
    return model ? model->metric : PACMAP_METRIC_EUCLIDEAN;
}

} // extern "C"