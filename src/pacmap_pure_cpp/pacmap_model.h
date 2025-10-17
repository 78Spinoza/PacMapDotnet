#pragma once

#include <vector>
#include <memory>
#include <random>
#include <string>
#include <cstdint>
#include <chrono>
#include <numeric>
#include "hnswlib.h"
#include "pcg_random.hpp"

#include "pacmap_simple_wrapper.h"

// Forward declarations for callbacks
typedef void (*pacmap_progress_callback_internal)(const char* phase, int current, int total, float percent, const char* message);

// Triplet types (PACMAP-specific)
enum TripletType {
    NEIGHBOR = 0,   // Nearest neighbor pairs (local structure)
    MID_NEAR = 1,   // Mid-near pairs (global structure)
    FURTHER = 2     // Far pairs (uniform distribution)
};

// Unified triplet structure (review-optimized) - FIX19: Use int64_t for large dataset support
struct Triplet {
    int64_t anchor;         // Anchor point index
    int64_t neighbor;       // Neighbor point index
    TripletType type;       // Type of triplet
    float weight = 1.0f;

    Triplet() : anchor(-1), neighbor(-1), type(NEIGHBOR), weight(1.0f) {}
    Triplet(int64_t a, int64_t n, TripletType t, float w = 1.0f) : anchor(a), neighbor(n), type(t), weight(w) {}
};


// Performance statistics for monitoring
struct PerformanceStats {
    double sampling_time_ms = 0.0;
    double optimization_time_ms = 0.0;
    size_t peak_memory_mb = 0;
    int total_triplets = 0;
    float final_loss = 0.0f;
    int iterations_completed = 0;

    // Performance tracking fields
    std::chrono::high_resolution_clock::time_point start_time;

    // Operation history tracking
    struct OperationRecord {
        std::string operation;
        double duration_ms;
        int64_t timestamp;
    };
    std::vector<OperationRecord> operation_history;
};

// Enhanced PACMAP Model Structure (Pure PACMAP Implementation) - FIX19: Large dataset support
struct PacMapModel {
    // Core PACMAP parameters - use int64_t for large dataset support (1M+ points)
    int64_t n_samples = 0;
    int64_t n_features = 0;
    int64_t n_components = 2;
    int64_t n_neighbors = 10;
    float mn_ratio = 0.5f;     // Mid-near pair ratio
    float fp_ratio = 2.0f;     // Far pair ratio
    float learning_rate = 1.0f; // Adam optimizer learning rate
    float initialization_std_dev = 1e-4f; // Standard deviation for embedding initialization (matches reference)
    int phase1_iters = 100;    // Global structure phase
    int phase2_iters = 100;    // Balance phase
    int phase3_iters = 250;    // Local structure phase
    PacMapMetric metric = PACMAP_METRIC_EUCLIDEAN;
    int random_seed = -1;      // -1: non-deterministic, else seeded

    // HNSW parameters for efficient nearest neighbor search
    int hnsw_m = 16;
    int hnsw_ef_construction = 200;
    int hnsw_ef_search = 200;
    bool use_quantization = false;

    // PACMAP algorithm state - use int64_t for large dataset support
    int64_t total_triplets = 0;
    int64_t neighbor_triplets = 0;
    int64_t mid_near_triplets = 0;
    int64_t far_triplets = 0;

    // Dual HNSW information for enhanced persistence
    uint32_t original_space_crc = 0;
    uint32_t embedding_space_crc = 0;
    uint32_t model_version_crc = 0;
    float hnsw_recall_percentage = 0.0f;
    bool always_save_embedding_data = false;

    // Data preprocessing fields for transform consistency - now double precision
    std::vector<double> feature_means;
    std::vector<double> feature_stds;

    // � CRITICAL FIX v2.8.4: Min-max normalization parameters (matching Python) - double precision
    double xmin = 0.0;  // Global minimum value for min-max scaling
    double xmax = 1.0;  // Global maximum value for min-max scaling

    // Distance percentiles for triplet filtering - double precision
    double p25_distance = 0.0;  // 25th percentile distance
    double p75_distance = 0.0;  // 75th percentile distance

    // Adam optimizer state - now double precision for numerical stability
    std::vector<double> adam_m;     // First moment vector (momentum) - double precision
    std::vector<double> adam_v;     // Second moment vector (RMSprop-like) - double precision
    float adam_beta1 = 0.9f;        // FIXED: Set to 0.9 to use Adam (Python reference) instead of Simple SGD
    float adam_beta2 = 0.999f;     // Adam beta2 parameter (RMSprop decay)
    float adam_eps = 1e-7f;        // Adam epsilon parameter for numerical stability (PYTHON FIX: matches Python reference exactly)

    // Factory fields for HNSW space creation
    std::unique_ptr<hnswlib::SpaceInterface<float>> original_space_factory;
    std::unique_ptr<hnswlib::SpaceInterface<float>> embedding_space_factory;

    //  Persistent metric spaces (fixes AccessViolationException from dangling pointers)
    std::unique_ptr<hnswlib::L2Space> original_space;
    std::unique_ptr<hnswlib::L2Space> embedding_space;

    // OVERFLOW FIX: Flat triplet storage using uint64_t for large dataset support
    // Format: [anchor1, neighbor1, type1, anchor2, neighbor2, type2, ...]
    std::vector<uint64_t> triplets_flat;  // 64-bit indexing prevents overflow with large datasets

    // Data storage - now double precision
    std::vector<double> training_data;
    std::vector<double> embedding;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> original_space_index;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> embedding_space_index;
    float median_original_distance = 0.0f;

    // Safety statistics (review addition) - double precision
    double min_embedding_dist = 0.0;
    double p95_embedding_dist = 0.0;

    // Additional embedding statistics for safety analysis - double precision
    double min_embedding_distance = 0.0;
    double p95_embedding_distance = 0.0;
    double p99_embedding_distance = 0.0;
    double mild_embedding_outlier_threshold = 0.0;
    double extreme_embedding_outlier_threshold = 0.0;
    double std_embedding_distance = 0.0;
    double mean_embedding_distance = 0.0;

    // Original space distance statistics for persistence - double precision
    double min_original_distance = 0.0;
    double mean_original_distance = 0.0;
    double std_original_distance = 0.0;
    double p95_original_distance = 0.0;
    double p99_original_distance = 0.0;
    double mild_original_outlier_threshold = 0.0;
    double extreme_original_outlier_threshold = 0.0;
    double exact_match_threshold = 0.0;

    // Additional persistence fields for transform data - double precision
    bool use_normalization = false;
    std::vector<int> nn_indices;
    std::vector<double> nn_distances;
    std::vector<double> nn_weights;

    // Quantization fields (PQ - Product Quantization)
    int pq_m = 0;
    std::vector<uint8_t> pq_codes;
    std::vector<float> pq_centroids;

    // PCG RNG for random operations (consistent everywhere)
    pcg64_fast rng;

    // Error handling (review addition)
    int last_error_code = 0;
    std::string last_error_message;

    // Performance tracking
    PerformanceStats performance_stats;

    // State tracking
    bool is_fitted = false;
    bool is_optimized = false;
    bool force_exact_knn = false; // Override flag to force brute-force k-NN

    // Constructor
    PacMapModel() {
        // Initialize PCG RNG with seed if provided
        if (random_seed >= 0) {
            rng.seed(random_seed);
        }
    }

    // OVERFLOW FIX: Helper functions for flat triplet storage with 64-bit indexing
    inline void add_triplet(uint64_t anchor, uint64_t neighbor, uint64_t type) {
        triplets_flat.push_back(anchor);
        triplets_flat.push_back(neighbor);
        triplets_flat.push_back(type);
    }

    inline uint64_t get_triplet_count() const {
        return static_cast<uint64_t>(triplets_flat.size() / 3);
    }

    inline void clear_triplets() {
        triplets_flat.clear();
    }

    // Destructor
    ~PacMapModel() = default;
};

// Utility functions
extern const char* pacmap_get_error_message(int error_code);
extern void set_last_error(PacMapModel* model, int error_code, const std::string& message);
extern int get_last_error_code(const PacMapModel* model);
extern const char* get_last_error_message(const PacMapModel* model);

// Model utility namespace for compatibility
namespace model_utils {
    PacMapModel* create_model();
    void destroy_model(PacMapModel* model);
    int get_model_info(PacMapModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, PacMapMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search);
    int get_model_info_v2(PacMapModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, PacMapMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search,
        uint32_t* original_crc, uint32_t* embedding_crc, uint32_t* version_crc,
        float* hnsw_recall_percentage);
    int get_embedding_dim(PacMapModel* model);
    int get_n_vertices(PacMapModel* model);
    int is_fitted(PacMapModel* model);
    const char* get_error_message(int error_code);
    const char* get_metric_name(PacMapMetric metric);
    const char* get_version();
}