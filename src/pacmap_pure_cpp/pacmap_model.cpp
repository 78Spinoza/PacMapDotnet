#include "pacmap_model.h"
#include "pacmap_progress_utils.h"

namespace model_utils {

    PacMapModel* create_model() {
        try {
            PacMapModel* model = new PacMapModel();
            return model;
        }
        catch (const std::bad_alloc&) {
            return nullptr;
        }
    }

    void destroy_model(PacMapModel* model) {
        if (model) {
            delete model;
        }
    }

    // Note: This function is kept for compatibility but uses PacMapModel structure
    int get_model_info(PacMapModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, PacMapMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search) {

        if (!model) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_samples; // Map from UMAP naming
        if (n_dim) *n_dim = model->n_features;         // Map from UMAP naming
        if (embedding_dim) *embedding_dim = model->n_components;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = 0.0f; // Not used in PACMAP
        if (spread) *spread = 1.0f;     // Not used in PACMAP
        if (metric) *metric = model->metric;
        if (hnsw_M) *hnsw_M = model->hnsw_m;
        if (hnsw_ef_construction) *hnsw_ef_construction = model->hnsw_ef_construction;
        if (hnsw_ef_search) *hnsw_ef_search = model->hnsw_ef_search;

        return PACMAP_SUCCESS;
    }

    int get_model_info_v2(PacMapModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, PacMapMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search,
        uint32_t* original_crc, uint32_t* embedding_crc, uint32_t* version_crc,
        float* hnsw_recall_percentage) {

        if (!model) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_samples;
        if (n_dim) *n_dim = model->n_features;
        if (embedding_dim) *embedding_dim = model->n_components;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->median_original_distance;
        if (spread) *spread = 1.0f; // Default spread
        if (metric) *metric = model->metric;
        if (hnsw_M) *hnsw_M = model->hnsw_m;
        if (hnsw_ef_construction) *hnsw_ef_construction = model->hnsw_ef_construction;
        if (hnsw_ef_search) *hnsw_ef_search = model->hnsw_ef_search;

        // New dual HNSW information
        if (original_crc) *original_crc = model->original_space_crc;
        if (embedding_crc) *embedding_crc = model->embedding_space_crc;
        if (version_crc) *version_crc = model->model_version_crc;
        if (hnsw_recall_percentage) *hnsw_recall_percentage = model->hnsw_recall_percentage;

        return PACMAP_SUCCESS;
    }

    int get_embedding_dim(PacMapModel* model) {
        if (!model) {
            return -1;
        }
        return model->n_components;
    }

    int get_n_vertices(PacMapModel* model) {
        if (!model) {
            return -1;
        }
        return model->n_samples;
    }

    int is_fitted(PacMapModel* model) {
        if (!model) {
            return 0;
        }
        return model->is_fitted ? 1 : 0;
    }

    const char* get_version() {
        return PACMAP_WRAPPER_VERSION_STRING;
    }

    const char* get_error_message(int error_code) {
        switch (error_code) {
        case PACMAP_SUCCESS:
            return "Success";
        case PACMAP_ERROR_INVALID_PARAMS:
            return "Invalid parameters provided";
        case PACMAP_ERROR_MEMORY:
            return "Memory allocation failed";
        case PACMAP_ERROR_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case PACMAP_ERROR_FILE_IO:
            return "File I/O operation failed";
        case PACMAP_ERROR_MODEL_NOT_FITTED:
            return "Model has not been fitted yet";
        case PACMAP_ERROR_INVALID_MODEL_FILE:
            return "Invalid model file";
        default:
            return "Unknown error";
        }
    }

    const char* get_metric_name(PacMapMetric metric) {
        switch (metric) {
        case PACMAP_METRIC_EUCLIDEAN:
            return "euclidean";
        case PACMAP_METRIC_COSINE:
            return "cosine";
        case PACMAP_METRIC_MANHATTAN:
            return "manhattan";
        case PACMAP_METRIC_CORRELATION:
            return "correlation";
        case PACMAP_METRIC_HAMMING:
            return "hamming";
        default:
            return "unknown";
        }
    }

  }