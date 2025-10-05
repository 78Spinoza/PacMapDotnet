#pragma once

#include <vector>
#include <memory>
#include <random>
#include <string>
#include <cstdint>
#include "hnswlib.h"

#include "pacmap_simple_wrapper.h"

// Forward declarations for callbacks
typedef void (*uwot_progress_callback_v2)(const char* phase, int current, int total, float percent, const char* message);

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
};

// Enhanced PACMAP Model Structure (review-optimized)
struct PacMapModel {
    // Core parameters
    int n_samples = 0;
    int n_features = 0;
    int n_components = 2;
    int n_neighbors = 10;
    float mn_ratio = 0.5f;
    float fp_ratio = 2.0f;
    float learning_rate = 1.0f;
    int phase1_iters = 100;
    int phase2_iters = 100;
    int phase3_iters = 250;
    PacMapMetric metric = PACMAP_METRIC_EUCLIDEAN;
    int random_seed = -1;  // -1: non-deterministic, else seeded

    // UMAP compatibility fields (for persistence)
    int n_vertices = 0;        // Alias for n_samples
    int n_dim = 0;            // Alias for n_features
    int embedding_dim = 2;    // Alias for n_components
    float min_dist = 0.0f;    // Not used in PACMAP but for compatibility
    float spread = 1.0f;      // Not used in PACMAP but for compatibility
    float a = 1.0f;           // UMAP curve parameters (for compatibility)
    float b = 1.0f;           // UMAP curve parameters (for compatibility)
    float mean_original_distance = 0.0f;  // For persistence compatibility

    // HNSW parameters (from UMAP)
    int hnsw_m = 16;
    int hnsw_M = 16;  // Alternative naming for compatibility
    int hnsw_ef_construction = 200;
    int hnsw_ef_search = 200;
    bool use_quantization = false;

    // Dual HNSW information for enhanced persistence
    uint32_t original_space_crc = 0;
    uint32_t embedding_space_crc = 0;
    uint32_t model_version_crc = 0;
    float hnsw_recall_percentage = 0.0f;
    bool always_save_embedding_data = false;

    // Additional UMAP compatibility fields
    int normalization_mode = 0;
    bool use_normalization = false;
    std::vector<int> nn_indices;
    std::vector<double> nn_distances;
    std::vector<double> nn_weights;

    // Unified triplet storage (review optimization)
    std::vector<Triplet> triplets;

    // Data storage
    std::vector<float> training_data;
    std::vector<float> embedding;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> original_space_index;

    // Safety statistics (review addition)
    float min_embedding_dist = 0.0f;
    float p95_embedding_dist = 0.0f;

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
}