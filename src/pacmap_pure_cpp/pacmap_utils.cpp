#include "pacmap_utils.h"
#include "pacmap_distance.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>

// Callback functions for progress reporting and warnings
static pacmap_progress_callback_v2 g_progress_callback = nullptr;

void send_warning_to_callback(const char* message) {
    if (g_progress_callback && message) {
        g_progress_callback("WARNING", 0, 0, 0.0f, message);
    } else {
        std::cerr << "WARNING: " << message << std::endl;
    }
}

void send_error_to_callback(const char* message) {
    if (g_progress_callback && message) {
        g_progress_callback("ERROR", 0, 0, 0.0f, message);
    } else {
        std::cerr << "ERROR: " << message << std::endl;
    }
}

void set_global_callback(pacmap_progress_callback_v2 callback) {
    g_progress_callback = callback;
}

int validate_parameters(PacMapModel* model) {
    if (!model) return PACMAP_ERROR_INVALID_PARAMS;

    // Validate basic parameters
    if (model->n_samples <= 0 || model->n_features <= 0) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMS,
                      "n_samples and n_features must be positive");
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    if (model->n_neighbors <= 0 || model->n_neighbors >= model->n_samples) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMS,
                      "n_neighbors must be between 1 and n_samples-1");
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    // Validate PACMAP-specific parameters
    if (model->mn_ratio < 0.0f || model->fp_ratio < 0.0f) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMS,
                      "MN_ratio and FP_ratio must be non-negative");
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    if (model->learning_rate <= 0.0f) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMS,
                      "learning_rate must be positive");
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    if (!is_valid_embedding_dimension(model->n_components)) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMS,
                      "embedding dimension must be between 2 and 100");
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    // Multi-metric support v2.8.24: Support all HNSW-compatible metrics
    if (!is_supported_metric(model->metric)) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMS,
                      "Unsupported metric specified. Supported metrics: euclidean, cosine, manhattan, hamming");
        return PACMAP_ERROR_INVALID_PARAMS;
    }

    // Issue warnings for metric-specific requirements
    if (model->metric == PACMAP_METRIC_HAMMING) {
        std::cout << "INFO: Hamming metric selected - binary conversion will be applied automatically." << std::endl;
    } else if (model->metric == PACMAP_METRIC_COSINE) {
        std::cout << "INFO: Cosine metric selected - L2 normalization will be applied automatically." << std::endl;
    }

    return PACMAP_SUCCESS;
}

void validate_metric_data(const float* data, int n_obs, int n_dim, PacMapMetric metric) {
    if (!data || n_obs <= 0 || n_dim <= 0) return;

    // Check for NaN/Inf values
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n_obs * n_dim; ++i) {
        if (std::isnan(data[i])) nan_count++;
        if (std::isinf(data[i])) inf_count++;
    }

    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "Warning: Data contains " << nan_count << " NaN and "
                  << inf_count << " Inf values" << std::endl;
    }

    // Metric-specific validation
    switch (metric) {
        case PACMAP_METRIC_HAMMING:
            // Hamming distance expects binary data
            for (int i = 0; i < n_obs * n_dim; ++i) {
                if (data[i] != 0.0f && data[i] != 1.0f) {
                    std::cerr << "Warning: Hamming metric expects binary data (0/1), found "
                              << data[i] << std::endl;
                }
            }
            break;
        case PACMAP_METRIC_CORRELATION:
            // Correlation requires variance > 0
            for (int j = 0; j < n_dim; ++j) {
                float mean = 0.0f;
                for (int i = 0; i < n_obs; ++i) {
                    mean += data[i * n_dim + j];
                }
                mean /= n_obs;

                float variance = 0.0f;
                for (int i = 0; i < n_obs; ++i) {
                    float diff = data[i * n_dim + j] - mean;
                    variance += diff * diff;
                }
                variance /= n_obs;

                if (variance < 1e-8f) {
                    std::cerr << "Warning: Feature " << j << " has near-zero variance, "
                              << "correlation metric may be unstable" << std::endl;
                }
            }
            break;
        default:
            break;
    }
}

bool check_memory_requirements(int n_samples, int n_features, int n_neighbors) {
    // Estimate memory requirements
    size_t data_size = n_samples * n_features * sizeof(float);
    size_t embedding_size = n_samples * 2 * sizeof(float);  // Assume 2D embedding
    size_t triplet_size = n_samples * n_neighbors * 3 * sizeof(Triplet);  // Rough estimate
    size_t hnsw_size = n_samples * n_neighbors * sizeof(int) * 2;  // Approximate HNSW memory

    size_t total_memory = data_size + embedding_size + triplet_size + hnsw_size;

    // Convert to MB and check against reasonable limit (e.g., 8GB)
    size_t memory_mb = total_memory / (1024 * 1024);
    const size_t max_memory_mb = 8 * 1024;  // 8GB limit

    if (memory_mb > max_memory_mb) {
        std::cerr << "Warning: Estimated memory usage (" << memory_mb
                  << " MB) exceeds recommended limit (" << max_memory_mb << " MB)" << std::endl;
        return false;
    }

    return true;
}

void auto_tune_parameters(PacMapModel* model, int n_samples) {
    if (n_samples < 100) {
        // Small dataset: more conservative parameters
        model->n_neighbors = std::min(model->n_neighbors, n_samples / 2);
        model->learning_rate = std::min(model->learning_rate, 0.5f);
        model->mn_ratio = std::max(model->mn_ratio, 0.2f);
    } else if (n_samples > 10000) {
        // Large dataset: more aggressive parameters
        model->n_neighbors = std::min(model->n_neighbors, 50);
        model->hnsw_ef_construction = std::min(model->hnsw_ef_construction, 100);
        model->use_quantization = true;
    }

    // Adjust iterations based on dataset size
    if (n_samples > 5000) {
        model->phase1_iters = std::min(model->phase1_iters, 50);
        model->phase2_iters = std::min(model->phase2_iters, 50);
        model->phase3_iters = std::min(model->phase3_iters, 100);
    }
}

bool detect_degenerate_cases(int n_samples, int n_features) {
    if (n_samples <= 1) {
        std::cerr << "Error: Need at least 2 samples for dimensionality reduction" << std::endl;
        return true;
    }

    if (n_features <= 1) {
        std::cerr << "Error: Need at least 2 features for meaningful dimensionality reduction" << std::endl;
        return true;
    }

    if (n_samples <= n_features) {
        std::cerr << "Warning: Number of samples (" << n_samples
                  << ") is less than or equal to number of features (" << n_features
                  << "). Results may be unstable." << std::endl;
    }

    return false;
}

bool check_for_nan_inf(const float* data, int size) {
    if (!data) return true;

    for (int i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            return true;
        }
    }
    return false;
}

// REMOVED: validate_triplet_distribution function - would need flat storage adaptation if needed

void start_performance_timer(PacMapModel* model) {
    model->performance_stats.start_time = std::chrono::high_resolution_clock::now();
}

void record_performance_stats(PacMapModel* model, const std::string& operation) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - model->performance_stats.start_time);

    PerformanceStats::OperationRecord record;
    record.operation = operation;
    record.duration_ms = duration.count();
    record.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        end_time.time_since_epoch()).count();

    model->performance_stats.operation_history.push_back(record);
}

PerformanceStats get_performance_stats(const PacMapModel* model) {
    return model->performance_stats;
}

float compute_distance(const float* x, const float* y, int n_features, PacMapMetric metric) {
    return distance_metrics::compute_distance(x, y, n_features, metric);
}

void normalize_data(std::vector<float>& data, int n_samples, int n_features, PacMapMetric metric) {
    if (data.empty()) return;

    if (metric == PACMAP_METRIC_COSINE) {
        // L2 normalization for cosine distance
        for (int i = 0; i < n_samples; ++i) {
            float norm = 0.0f;
            for (int j = 0; j < n_features; ++j) {
                float val = data[i * n_features + j];
                norm += val * val;
            }
            norm = std::sqrt(norm);

            if (norm > 1e-8f) {
                for (int j = 0; j < n_features; ++j) {
                    data[i * n_features + j] /= norm;
                }
            }
        }
    } else if (metric == PACMAP_METRIC_CORRELATION) {
        // Z-score normalization for correlation distance
        for (int j = 0; j < n_features; ++j) {
            // Compute mean and standard deviation
            float mean = 0.0f;
            for (int i = 0; i < n_samples; ++i) {
                mean += data[i * n_features + j];
            }
            mean /= n_samples;

            float std_dev = 0.0f;
            for (int i = 0; i < n_samples; ++i) {
                float diff = data[i * n_features + j] - mean;
                std_dev += diff * diff;
            }
            std_dev = std::sqrt(std_dev / n_samples);

            // Normalize feature
            if (std_dev > 1e-8f) {
                for (int i = 0; i < n_samples; ++i) {
                    data[i * n_features + j] = (data[i * n_features + j] - mean) / std_dev;
                }
            }
        }
    }
}

void standardize_data(std::vector<float>& data, int n_samples, int n_features) {
    if (data.empty()) return;

    // Global standardization: zero mean, unit variance
    float global_mean = 0.0f;
    for (float val : data) {
        global_mean += val;
    }
    global_mean /= data.size();

    float global_var = 0.0f;
    for (float val : data) {
        float diff = val - global_mean;
        global_var += diff * diff;
    }
    global_var /= data.size();
    float global_std = std::sqrt(global_var);

    if (global_std > 1e-8f) {
        for (float& val : data) {
            val = (val - global_mean) / global_std;
        }
    }
}

std::mt19937 get_seeded_mt19937(int seed) {
    return seed >= 0 ? std::mt19937(seed) : std::mt19937(std::random_device{}());
}

void set_random_seed(PacMapModel* model, int seed) {
    model->random_seed = seed;
    model->rng = get_seeded_mt19937(seed);
}

void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

std::string format_duration(double milliseconds) {
    if (milliseconds < 1000.0) {
        return std::to_string(static_cast<int>(milliseconds)) + " ms";
    } else if (milliseconds < 60000.0) {
        return std::to_string(static_cast<int>(milliseconds / 1000.0)) + " s";
    } else {
        int minutes = static_cast<int>(milliseconds / 60000.0);
        int seconds = static_cast<int>((milliseconds - minutes * 60000.0) / 1000.0);
        return std::to_string(minutes) + " min " + std::to_string(seconds) + " s";
    }
}

bool is_valid_parameter_combination(float mn_ratio, float fp_ratio, int n_neighbors) {
    if (n_neighbors <= 0) return false;

    // Check if ratios would result in reasonable triplet counts
    int mn_triplets = static_cast<int>(n_neighbors * mn_ratio);
    int fp_triplets = static_cast<int>(n_neighbors * fp_ratio);

    if (mn_triplets < 1 || fp_triplets < 1) {
        return false;  // Need at least some triplets of each type
    }

    // Check for extreme ratios
    if (mn_ratio > 10.0f || fp_ratio > 10.0f) {
        return false;  // Very large ratios may cause performance issues
    }

    return true;
}

bool is_supported_metric(PacMapMetric metric) {
    switch (metric) {
        case PACMAP_METRIC_EUCLIDEAN:
        case PACMAP_METRIC_COSINE:
        case PACMAP_METRIC_MANHATTAN:
        case PACMAP_METRIC_CORRELATION:
        case PACMAP_METRIC_HAMMING:
            return true;
        default:
            return false;
    }
}

bool is_valid_embedding_dimension(int n_components) {
    return n_components >= 2 && n_components <= 100;
}

void set_last_error(PacMapModel* model, int error_code, const std::string& message) {
    model->last_error_code = error_code;
    model->last_error_message = message;
}

const char* internal_pacmap_get_error_message(int error_code) {
    switch (error_code) {
        case PACMAP_SUCCESS: return "Success";
        case PACMAP_ERROR_INVALID_PARAMS: return "Invalid parameters provided";
        case PACMAP_ERROR_MEMORY: return "Memory allocation failed";
        case PACMAP_ERROR_NOT_IMPLEMENTED: return "Feature not implemented";
        case PACMAP_ERROR_FILE_IO: return "File I/O operation failed";
        case PACMAP_ERROR_MODEL_NOT_FITTED: return "Model has not been fitted yet";
        case PACMAP_ERROR_INVALID_MODEL_FILE: return "Invalid model file";
        case PACMAP_ERROR_CRC_MISMATCH: return "CRC32 validation failed - file corruption";
        default: return "Unknown error";
    }
}

int get_last_error_code(PacMapModel* model) {
    return model ? model->last_error_code : PACMAP_ERROR_INVALID_PARAMS;
}

const char* get_last_error_message(PacMapModel* model) {
    return model ? model->last_error_message.c_str() : "Invalid model pointer";
}

void print_model_info(const PacMapModel* model) {
    if (!model) {
        std::cout << "Model: null" << std::endl;
        return;
    }

    std::cout << "=== PACMAP Model Information ===" << std::endl;
    std::cout << "Samples: " << model->n_samples << std::endl;
    std::cout << "Features: " << model->n_features << std::endl;
    std::cout << "Embedding Dim: " << model->n_components << std::endl;
    std::cout << "Neighbors: " << model->n_neighbors << std::endl;
    std::cout << "MN Ratio: " << model->mn_ratio << std::endl;
    std::cout << "FP Ratio: " << model->fp_ratio << std::endl;
    std::cout << "Learning Rate: " << model->learning_rate << std::endl;
    std::cout << "Phase Iterations: (" << model->phase1_iters << ", "
              << model->phase2_iters << ", " << model->phase3_iters << ")" << std::endl;
    std::cout << "Random Seed: " << model->random_seed << std::endl;
    std::cout << "Metric: " << static_cast<int>(model->metric) << std::endl;
    std::cout << "Triplets: " << model->get_triplet_count() << std::endl;
    std::cout << "===============================" << std::endl;
}

// REMOVED: print_triplet_stats function - would need flat storage adaptation if needed

void print_performance_stats(const PerformanceStats& stats) {
    std::cout << "=== Performance Statistics ===" << std::endl;

    if (stats.operation_history.empty()) {
        std::cout << "No operations recorded" << std::endl;
        return;
    }

    double total_time = 0.0;
    for (const auto& op : stats.operation_history) {
        total_time += op.duration_ms;
        std::cout << op.operation << ": " << op.duration_ms << " ms" << std::endl;
    }

    std::cout << "Total Time: " << total_time << " ms" << std::endl;
    std::cout << "Operations: " << stats.operation_history.size() << std::endl;
    std::cout << "Average Time: " << (total_time / stats.operation_history.size()) << " ms" << std::endl;
    std::cout << "==============================" << std::endl;
}