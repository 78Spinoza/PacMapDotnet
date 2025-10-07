#pragma once

#include <vector>
#include <memory>
#include <random>
#include <string>
#include <cstdint>
#include <chrono>
#include "hnswlib.h"

#include "pacmap_simple_wrapper.h"

// Forward declarations for callbacks
typedef void (*pacmap_progress_callback_internal)(const char* phase, int current, int total, float percent, const char* message);

// Triplet types (PACMAP-specific)
enum TripletType {
    NEIGHBOR = 0,   // Nearest neighbor pairs (local structure)
    MID_NEAR = 1,   // Mid-near pairs (global structure)
    FURTHER = 2     // Far pairs (uniform distribution)
};

// Unified triplet structure (review-optimized)
struct Triplet {
    int anchor;         // Anchor point index
    int neighbor;       // Neighbor point index
    TripletType type;   // Type of triplet
    float weight = 1.0f;

    Triplet() : anchor(-1), neighbor(-1), type(NEIGHBOR), weight(1.0f) {}
    Triplet(int a, int n, TripletType t, float w = 1.0f) : anchor(a), neighbor(n), type(t), weight(w) {}
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

// Enhanced PACMAP Model Structure (Pure PACMAP Implementation)
struct PacMapModel {
    // Core PACMAP parameters
    int n_samples = 0;
    int n_features = 0;
    int n_components = 2;
    int n_neighbors = 10;
    float mn_ratio = 0.5f;     // Mid-near pair ratio
    float fp_ratio = 2.0f;     // Far pair ratio
    float learning_rate = 1.0f; // Adam optimizer learning rate
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

    // PACMAP algorithm state
    int total_triplets = 0;
    int neighbor_triplets = 0;
    int mid_near_triplets = 0;
    int far_triplets = 0;

    // Dual HNSW information for enhanced persistence
    uint32_t original_space_crc = 0;
    uint32_t embedding_space_crc = 0;
    uint32_t model_version_crc = 0;
    float hnsw_recall_percentage = 0.0f;
    bool always_save_embedding_data = false;

    // Data preprocessing fields for transform consistency
    std::vector<float> feature_means;
    std::vector<float> feature_stds;

    // Distance percentiles for triplet filtering
    float p25_distance = 0.0f;  // 25th percentile distance
    float p75_distance = 0.0f;  // 75th percentile distance

    // Adam optimizer state
    std::vector<float> adam_m;  // First moment vector
    std::vector<float> adam_v;  // Second moment vector
    float adam_beta1 = 0.9f;    // Adam beta1 parameter
    float adam_beta2 = 0.999f;  // Adam beta2 parameter
    float adam_eps = 1e-8f;     // Adam epsilon parameter

    // Factory fields for HNSW space creation
    std::unique_ptr<hnswlib::SpaceInterface<float>> original_space_factory;
    std::unique_ptr<hnswlib::SpaceInterface<float>> embedding_space_factory;

    // Unified triplet storage (review optimization)
    std::vector<Triplet> triplets;

    // Data storage
    std::vector<float> training_data;
    std::vector<float> embedding;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> original_space_index;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> embedding_space_index;
    float median_original_distance = 0.0f;

    // Safety statistics (review addition)
    float min_embedding_dist = 0.0f;
    float p95_embedding_dist = 0.0f;

    // Additional embedding statistics for safety analysis
    float min_embedding_distance = 0.0f;
    float p95_embedding_distance = 0.0f;
    float p99_embedding_distance = 0.0f;
    float mild_embedding_outlier_threshold = 0.0f;
    float extreme_embedding_outlier_threshold = 0.0f;
    float std_embedding_distance = 0.0f;
    float mean_embedding_distance = 0.0f;

    // Original space distance statistics for persistence
    float min_original_distance = 0.0f;
    float mean_original_distance = 0.0f;
    float std_original_distance = 0.0f;
    float p95_original_distance = 0.0f;
    float p99_original_distance = 0.0f;
    float mild_original_outlier_threshold = 0.0f;
    float extreme_original_outlier_threshold = 0.0f;
    float exact_match_threshold = 0.0f;

    // Additional persistence fields for transform data
    bool use_normalization = false;
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    std::vector<float> nn_weights;

    // Quantization fields (PQ - Product Quantization)
    int pq_m = 0;
    std::vector<uint8_t> pq_codes;
    std::vector<float> pq_centroids;

    // RNG for deterministic behavior (review requirement)
    std::mt19937 rng;

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
    PacMapModel() = default;

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