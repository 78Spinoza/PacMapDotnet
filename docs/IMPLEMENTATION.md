# PACMAPCSharp Implementation Guide

## Overview

This document provides detailed step-by-step instructions for implementing PACMAP in C++, based on the Python reference implementation and the existing UMAP infrastructure.

## C++ Implementation Details

### 1. Data Structures

#### 1.1 Replace UMAP Structures with PACMAP Structures

```cpp
// File: pacmap_model.h

#pragma once

#include <vector>
#include <memory>
#include <tuple>

// PACMAP Triplet Types
enum TripletType {
    NEIGHBOR = 0,    // Nearest neighbor pairs (local structure)
    MID_NEAR = 1,    // Mid-near pairs (global structure)
    FURTHER = 2      // Far pairs (uniform distribution)
};

// PACMAP Triplet Structure
struct Triplet {
    int anchor_idx;      // Anchor point index
    int neighbor_idx;    // Neighbor point index
    TripletType type;    // Type of triplet
    float weight;        // Dynamic weight based on optimization phase

    Triplet() : anchor_idx(-1), neighbor_idx(-1), type(NEIGHBOR), weight(1.0f) {}
    Triplet(int anchor, int neighbor, TripletType t, float w = 1.0f)
        : anchor_idx(anchor), neighbor_idx(neighbor), type(t), weight(w) {}
};

// PACMAP Model Structure (replaces UwotModel)
struct PacMapModel {
    // Training parameters
    int n_samples;
    int n_features;
    int n_components;
    int n_neighbors;
    float MN_ratio;
    float FP_ratio;
    float learning_rate;
    std::tuple<int, int, int> num_iters;  // (phase1, phase2, phase3)

    // Triplet sets
    std::vector<Triplet> neighbor_triplets;
    std::vector<Triplet> mid_near_triplets;
    std::vector<Triplet> further_triplets;

    // HNSW optimization (keep from UMAP)
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> ann_index;

    // Training data (optional, for transforms)
    std::vector<float> training_data;

    // Distance metric
    int distance_metric;

    // Random seed
    int random_seed;

    // Constructor
    PacMapModel() : n_samples(0), n_features(0), n_components(2),
                   n_neighbors(10), MN_ratio(0.5f), FP_ratio(2.0f),
                   learning_rate(1.0f), num_iters(std::make_tuple(100, 100, 250)),
                   distance_metric(0), random_seed(-1) {}
};
```

#### 1.2 Distance Metrics (reuse from UMAP)

```cpp
// File: pacmap_distance.h
// Reuse existing distance functions from UMAP
// - euclid_dist()
// - manhattan_dist()
// - angular_dist()
// - hamming_dist()
```

### 2. Core Algorithm Functions

#### 2.1 Triplet Sampling Functions

```cpp
// File: pacmap_triplet_sampling.h

#pragma once

#include "pacmap_model.h"
#include <random>
#include <algorithm>

// Neighbor pair sampling using HNSW
void sample_neighbors_pair(float* data, int n_samples, int n_dims,
                          int n_neighbors, std::vector<Triplet>& triplets,
                          hnswlib::HierarchicalNSW<float>* ann_index);

// Mid-near pair sampling for global structure
void sample_MN_pair(float* data, int n_samples, int n_dims,
                   int n_MN, std::vector<Triplet>& triplets,
                   int random_seed = -1);

// Far pair sampling for uniform distribution
void sample_FP_pair(float* data, int n_samples, int n_dims,
                   int n_FP, std::vector<Triplet>& triplets,
                   int random_seed = -1);

// Helper functions
void distance_based_sampling(float* data, int n_samples, int n_dims,
                           int target_pairs, float min_dist, float max_dist,
                           std::vector<Triplet>& triplets, int random_seed);
```

```cpp
// File: pacmap_triplet_sampling.cpp

#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include <random>
#include <unordered_set>
#include <numeric>

void sample_neighbors_pair(float* data, int n_samples, int n_dims,
                          int n_neighbors, std::vector<Triplet>& triplets,
                          hnswlib::HierarchicalNSW<float>* ann_index) {

    triplets.clear();
    triplets.reserve(n_samples * n_neighbors);

    // Use HNSW for fast neighbor search
    for (int i = 0; i < n_samples; i++) {
        // Find k+1 nearest neighbors (includes self)
        auto* query_labels = new int[n_neighbors + 1];
        auto* query_distances = new float[n_neighbors + 1];

        ann_index->searchKnn(data + i * n_dims, n_neighbors + 1,
                           query_labels, query_distances);

        // Skip self (first result) and add neighbors
        for (int j = 1; j <= n_neighbors; j++) {
            int neighbor_idx = query_labels[j];
            triplets.emplace_back(i, neighbor_idx, NEIGHBOR, 1.0f);
        }

        delete[] query_labels;
        delete[] query_distances;
    }
}

void sample_MN_pair(float* data, int n_samples, int n_dims,
                   int n_MN, std::vector<Triplet>& triplets,
                   int random_seed) {

    triplets.clear();
    triplets.reserve(n_samples * n_MN);

    std::mt19937 rng(random_seed >= 0 ? random_seed : std::random_device{}());
    std::uniform_int_distribution<int> uniform_dist(0, n_samples - 1);

    // Calculate distance matrix statistics
    std::vector<float> all_distances;
    for (int i = 0; i < std::min(n_samples, 1000); i++) {
        for (int j = i + 1; j < std::min(n_samples, 1000); j++) {
            float dist = euclid_dist(data + i * n_dims, data + j * n_dims, n_dims);
            all_distances.push_back(dist);
        }
    }

    // Determine mid-near distance range (25th-75th percentile)
    std::sort(all_distances.begin(), all_distances.end());
    float min_mn_dist = all_distances[all_distances.size() * 0.25];
    float max_mn_dist = all_distances[all_distances.size() * 0.75];

    distance_based_sampling(data, n_samples, n_dims, n_MN * n_samples,
                           min_mn_dist, max_mn_dist, triplets, random_seed);
}

void sample_FP_pair(float* data, int n_samples, int n_dims,
                   int n_FP, std::vector<Triplet>& triplets,
                   int random_seed) {

    triplets.clear();
    triplets.reserve(n_samples * n_FP);

    // Calculate far distance range (90th-100th percentile)
    std::vector<float> all_distances;
    for (int i = 0; i < std::min(n_samples, 500); i++) {
        for (int j = i + 1; j < std::min(n_samples, 500); j++) {
            float dist = euclid_dist(data + i * n_dims, data + j * n_dims, n_dims);
            all_distances.push_back(dist);
        }
    }

    std::sort(all_distances.begin(), all_distances.end());
    float min_fp_dist = all_distances[all_distances.size() * 0.9];

    distance_based_sampling(data, n_samples, n_dims, n_FP * n_samples,
                           min_fp_dist, std::numeric_limits<float>::infinity(),
                           triplets, random_seed);
}

void distance_based_sampling(float* data, int n_samples, int n_dims,
                           int target_pairs, float min_dist, float max_dist,
                           std::vector<Triplet>& triplets, int random_seed) {

    std::mt19937 rng(random_seed >= 0 ? random_seed : std::random_device{}());
    std::uniform_int_distribution<int> uniform_dist(0, n_samples - 1);

    std::unordered_set<long long> used_pairs;
    int pairs_found = 0;

    while (pairs_found < target_pairs) {
        int i = uniform_dist(rng);
        int j = uniform_dist(rng);

        if (i == j) continue;

        // Ensure unique pairs
        long long pair_key = ((long long)std::min(i, j) << 32) | std::max(i, j);
        if (used_pairs.find(pair_key) != used_pairs.end()) continue;

        float dist = euclid_dist(data + i * n_dims, data + j * n_dims, n_dims);

        if (dist >= min_dist && dist <= max_dist) {
            TripletType type = (max_dist == std::numeric_limits<float>::infinity()) ? FURTHER : MID_NEAR;
            triplets.emplace_back(i, j, type, 1.0f);
            used_pairs.insert(pair_key);
            pairs_found++;
        }
    }
}
```

#### 2.2 Weight Schedule Function

```cpp
// File: pacmap_optimization.h

#pragma once

#include <tuple>

// Three-phase weight schedule
std::tuple<float, float, float> find_weight(int iter, int total_iters);

// Optimization parameters
struct OptimizationWeights {
    float w_neighbors;
    float w_MN;
    float w_FP;
};
```

```cpp
// File: pacmap_optimization.cpp

#include "pacmap_optimization.h"

std::tuple<float, float, float> find_weight(int iter, int total_iters) {
    float progress = static_cast<float>(iter) / static_cast<float>(total_iters);

    float w_neighbors, w_MN, w_FP;

    if (progress < 0.1f) {
        // Phase 1: Global structure focus (0-10%)
        // w_MN decreases from 1000 to 3
        w_MN = 1000.0f * (1.0f - progress * 10.0f) + 3.0f * (progress * 10.0f);
        w_neighbors = 1.0f;
        w_FP = 1.0f;
    }
    else if (progress < 0.4f) {
        // Phase 2: Balance phase (10-40%)
        w_MN = 3.0f;
        w_neighbors = 1.0f;
        w_FP = 1.0f;
    }
    else {
        // Phase 3: Local structure focus (40-100%)
        // w_MN decreases from 3 to 0
        w_MN = 3.0f * (1.0f - (progress - 0.4f) / 0.6f);
        w_neighbors = 1.0f;
        w_FP = 1.0f;
    }

    return std::make_tuple(w_neighbors, w_MN, w_FP);
}
```

#### 2.3 Gradient Computation

```cpp
// File: pacmap_gradient.h

#pragma once

#include "pacmap_model.h"
#include "pacmap_optimization.h"

// PACMAP gradient computation
void pacmap_grad(float* embedding, std::vector<Triplet>& triplets,
                float w_neighbors, float w_MN, float w_FP,
                float* gradients, int n_samples, int n_components);

// Embedding update function
void update_embedding(float* embedding, float* gradients,
                     int n_samples, int n_components, float learning_rate);

// Loss function computation (for monitoring)
float compute_pacmap_loss(float* embedding, std::vector<Triplet>& triplets,
                         float w_neighbors, float w_MN, float w_FP,
                         int n_components);
```

```cpp
// File: pacmap_gradient.cpp

#include "pacmap_gradient.h"
#include "pacmap_distance.h"
#include <cmath>

void pacmap_grad(float* embedding, std::vector<Triplet>& triplets,
                float w_neighbors, float w_MN, float w_FP,
                float* gradients, int n_samples, int n_components) {

    // Initialize gradients to zero
    std::fill(gradients, gradients + n_samples * n_components, 0.0f);

    // Compute gradients for each triplet
    for (const auto& triplet : triplets) {
        int i = triplet.anchor_idx;
        int j = triplet.neighbor_idx;
        TripletType type = triplet.type;

        // Compute distance in embedding space
        float* emb_i = embedding + i * n_components;
        float* emb_j = embedding + j * n_components;

        float diff_squared = 0.0f;
        for (int d = 0; d < n_components; d++) {
            float diff = emb_i[d] - emb_j[d];
            diff_squared += diff * diff;
        }

        float d_ij = std::sqrt(diff_squared);

        // Avoid numerical issues
        if (d_ij < 1e-8f) continue;

        // Compute gradient magnitude based on triplet type
        float grad_magnitude;
        switch (type) {
            case NEIGHBOR:
                // Pull closer: w * 10 / ((10 + d)^2)
                grad_magnitude = w_neighbors * 10.0f / ((10.0f + d_ij) * (10.0f + d_ij));
                break;
            case MID_NEAR:
                // Moderate pull: w * 10000 / ((10000 + d)^2)
                grad_magnitude = w_MN * 10000.0f / ((10000.0f + d_ij) * (10000.0f + d_ij));
                break;
            case FURTHER:
                // Push apart: -w / ((1 + d)^2)
                grad_magnitude = -w_FP / ((1.0f + d_ij) * (1.0f + d_ij));
                break;
        }

        // Apply gradient (symmetric for both points)
        for (int d = 0; d < n_components; d++) {
            float diff = emb_i[d] - emb_j[d];
            float gradient = grad_magnitude * diff / d_ij;

            gradients[i * n_components + d] += gradient;
            gradients[j * n_components + d] -= gradient;
        }
    }
}

void update_embedding(float* embedding, float* gradients,
                     int n_samples, int n_components, float learning_rate) {

    // Simple gradient descent update
    for (int i = 0; i < n_samples * n_components; i++) {
        embedding[i] -= learning_rate * gradients[i];
    }
}

float compute_pacmap_loss(float* embedding, std::vector<Triplet>& triplets,
                         float w_neighbors, float w_MN, float w_FP,
                         int n_components) {

    float total_loss = 0.0f;

    for (const auto& triplet : triplets) {
        int i = triplet.anchor_idx;
        int j = triplet.neighbor_idx;
        TripletType type = triplet.type;

        // Compute distance in embedding space
        float* emb_i = embedding + i * n_components;
        float* emb_j = embedding + j * n_components;

        float d_ij = euclid_dist(emb_i, emb_j, n_components);

        // Compute loss based on triplet type
        float loss;
        switch (type) {
            case NEIGHBOR:
                loss = w_neighbors * (d_ij / (10.0f + d_ij));
                break;
            case MID_NEAR:
                loss = w_MN * (d_ij / (10000.0f + d_ij));
                break;
            case FURTHER:
                loss = w_FP * (1.0f / (1.0f + d_ij));
                break;
        }

        total_loss += loss;
    }

    return total_loss;
}
```

#### 2.4 Main Optimization Loop

```cpp
// File: pacmap_fit.h

#pragma once

#include "pacmap_model.h"

// Main PACMAP fitting function
int pacmap_fit(PacMapModel* model, float* data, int n_samples, int n_dims,
               int n_components, int n_neighbors, float MN_ratio, float FP_ratio,
               int num_iters_phase1, int num_iters_phase2, int num_iters_phase3,
               int distance_metric, float learning_rate,
               float* embedding, int random_seed);

// Helper functions
void initialize_random_embedding(float* embedding, int n_samples, int n_components, int random_seed);
```

```cpp
// File: pacmap_fit.cpp

#include "pacmap_fit.h"
#include "pacmap_triplet_sampling.h"
#include "pacmap_gradient.h"
#include "pacmap_optimization.h"
#include "pacmap_hnsw_utils.h"  // Reuse HNSW utilities from UMAP
#include <random>
#include <iostream>
#include <iomanip>

int pacmap_fit(PacMapModel* model, float* data, int n_samples, int n_dims,
               int n_components, int n_neighbors, float MN_ratio, float FP_ratio,
               int num_iters_phase1, int num_iters_phase2, int num_iters_phase3,
               int distance_metric, float learning_rate,
               float* embedding, int random_seed) {

    // Set up model parameters
    model->n_samples = n_samples;
    model->n_features = n_dims;
    model->n_components = n_components;
    model->n_neighbors = n_neighbors;
    model->MN_ratio = MN_ratio;
    model->FP_ratio = FP_ratio;
    model->learning_rate = learning_rate;
    model->num_iters = std::make_tuple(num_iters_phase1, num_iters_phase2, num_iters_phase3);
    model->distance_metric = distance_metric;
    model->random_seed = random_seed;

    // Step 1: Initialize embedding randomly
    initialize_random_embedding(embedding, n_samples, n_components, random_seed);

    // Step 2: Build HNSW index for fast neighbor search
    // Reuse HNSW infrastructure from UMAP
    model->ann_index = create_hnsw_index(data, n_samples, n_dims, distance_metric);

    // Step 3: Sample triplets
    std::vector<Triplet> neighbor_triplets, mid_near_triplets, further_triplets;

    std::cout << "Sampling neighbor pairs..." << std::endl;
    sample_neighbors_pair(data, n_samples, n_dims, n_neighbors,
                         neighbor_triplets, model->ann_index.get());

    std::cout << "Sampling mid-near pairs..." << std::endl;
    int n_MN = static_cast<int>(n_neighbors * MN_ratio);
    sample_MN_pair(data, n_samples, n_dims, n_MN, mid_near_triplets, random_seed);

    std::cout << "Sampling far pairs..." << std::endl;
    int n_FP = static_cast<int>(n_neighbors * FP_ratio);
    sample_FP_pair(data, n_samples, n_dims, n_FP, further_triplets, random_seed);

    // Store triplets in model
    model->neighbor_triplets = neighbor_triplets;
    model->mid_near_triplets = mid_near_triplets;
    model->further_triplets = further_triplets;

    // Step 4: Combine all triplets
    std::vector<Triplet> all_triplets;
    all_triplets.insert(all_triplets.end(), neighbor_triplets.begin(), neighbor_triplets.end());
    all_triplets.insert(all_triplets.end(), mid_near_triplets.begin(), mid_near_triplets.end());
    all_triplets.insert(all_triplets.end(), further_triplets.begin(), further_triplets.end());

    std::cout << "Total triplets: " << all_triplets.size() << std::endl;
    std::cout << "  Neighbor pairs: " << neighbor_triplets.size() << std::endl;
    std::cout << "  Mid-near pairs: " << mid_near_triplets.size() << std::endl;
    std::cout << "  Far pairs: " << further_triplets.size() << std::endl;

    // Step 5: Optimize embedding using three-phase gradient descent
    int total_iters = num_iters_phase1 + num_iters_phase2 + num_iters_phase3;

    std::cout << "Starting optimization..." << std::endl;
    std::cout << "Phase 1: " << num_iters_phase1 << " iterations (global structure)" << std::endl;
    std::cout << "Phase 2: " << num_iters_phase2 << " iterations (balance)" << std::endl;
    std::cout << "Phase 3: " << num_iters_phase3 << " iterations (local structure)" << std::endl;

    // Allocate gradients
    std::vector<float> gradients(n_samples * n_components);

    for (int iter = 0; iter < total_iters; iter++) {
        // Get weights for current iteration
        auto [w_neighbors, w_MN, w_FP] = find_weight(iter, total_iters);

        // Compute gradients
        pacmap_grad(embedding, all_triplets, w_neighbors, w_MN, w_FP,
                   gradients.data(), n_samples, n_components);

        // Update embedding
        update_embedding(embedding, gradients.data(), n_samples, n_components, learning_rate);

        // Print progress
        if (iter % 100 == 0) {
            float loss = compute_pacmap_loss(embedding, all_triplets, w_neighbors, w_MN, w_FP, n_components);
            std::cout << "Iter " << std::setw(4) << iter << "/" << total_iters
                      << " | Loss: " << std::scientific << std::setprecision(4) << loss
                      << " | Weights: w_n=" << w_neighbors << ", w_mn=" << w_MN << ", w_fp=" << w_FP << std::endl;
        }
    }

    std::cout << "Optimization completed." << std::endl;

    // Optional: Store training data for transforms
    model->training_data.assign(data, data + n_samples * n_dims);

    return 0; // Success
}

void initialize_random_embedding(float* embedding, int n_samples, int n_components, int random_seed) {
    std::mt19937 rng(random_seed >= 0 ? random_seed : std::random_device{}());
    std::normal_distribution<float> normal_dist(0.0f, 1e-4f);

    for (int i = 0; i < n_samples * n_components; i++) {
        embedding[i] = normal_dist(rng);
    }
}
```

### 3. C++ API Wrapper

#### 3.1 Update pacmap_simple_wrapper.h

```cpp
// File: pacmap_simple_wrapper.h

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// PACMAP model handle (opaque pointer)
typedef struct PacMapModel PacMapModel;

// Distance metrics (reuse from UMAP)
typedef enum {
    PACMAP_METRIC_EUCLIDEAN = 0,
    PACMAP_METRIC_COSINE = 1,
    PACMAP_METRIC_MANHATTAN = 2,
    PACMAP_METRIC_CORRELATION = 3,
    PACMAP_METRIC_HAMMING = 4
} PacMapMetric;

// Progress callback
typedef void (*pacmap_progress_callback)(int current, int total, float percent, const char* message);

// Model management
PacMapModel* pacmap_create_model();
void pacmap_destroy_model(PacMapModel* model);

// Main fitting function
int pacmap_fit(PacMapModel* model,
               float* data, int n_samples, int n_features,
               int n_components, int n_neighbors, float MN_ratio, float FP_ratio,
               int num_iters_phase1, int num_iters_phase2, int num_iters_phase3,
               PacMapMetric metric, float learning_rate,
               float* embedding, int random_seed);

// Transform function (for new data points)
int pacmap_transform(PacMapModel* model, float* new_data, int n_new,
                     float* embedding);

// Model persistence
int pacmap_save_model(PacMapModel* model, const char* filename);
PacMapModel* pacmap_load_model(const char* filename);

// Model information
int pacmap_get_model_info(PacMapModel* model,
                         int* n_samples, int* n_features, int* n_components,
                         int* n_neighbors, float* MN_ratio, float* FP_ratio,
                         PacMapMetric* metric);

#ifdef __cplusplus
}
#endif
```

#### 3.2 Update pacmap_simple_wrapper.cpp

```cpp
// File: pacmap_simple_wrapper.cpp

#include "pacmap_simple_wrapper.h"
#include "pacmap_fit.h"
#include "pacmap_transform.h"
#include "pacmap_persistence.h"
#include <iostream>

extern "C" {

PacMapModel* pacmap_create_model() {
    return new PacMapModel();
}

void pacmap_destroy_model(PacMapModel* model) {
    delete model;
}

int pacmap_fit(PacMapModel* model,
               float* data, int n_samples, int n_features,
               int n_components, int n_neighbors, float MN_ratio, float FP_ratio,
               int num_iters_phase1, int num_iters_phase2, int num_iters_phase3,
               PacMapMetric metric, float learning_rate,
               float* embedding, int random_seed) {

    if (!model || !data || !embedding) {
        return -1; // Invalid parameters
    }

    try {
        return pacmap_fit(model, data, n_samples, n_features,
                         n_components, n_neighbors, MN_ratio, FP_ratio,
                         num_iters_phase1, num_iters_phase2, num_iters_phase3,
                         static_cast<int>(metric), learning_rate,
                         embedding, random_seed);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in pacmap_fit: " << e.what() << std::endl;
        return -2;
    }
}

int pacmap_transform(PacMapModel* model, float* new_data, int n_new, float* embedding) {
    if (!model || !new_data || !embedding) {
        return -1;
    }

    // Implementation would go here
    // For now, return not implemented
    return -3;
}

int pacmap_save_model(PacMapModel* model, const char* filename) {
    if (!model || !filename) {
        return -1;
    }

    // Implementation would go here
    return -3;
}

PacMapModel* pacmap_load_model(const char* filename) {
    if (!filename) {
        return nullptr;
    }

    // Implementation would go here
    return nullptr;
}

int pacmap_get_model_info(PacMapModel* model,
                         int* n_samples, int* n_features, int* n_components,
                         int* n_neighbors, float* MN_ratio, float* FP_ratio,
                         PacMapMetric* metric) {
    if (!model) {
        return -1;
    }

    if (n_samples) *n_samples = model->n_samples;
    if (n_features) *n_features = model->n_features;
    if (n_components) *n_components = model->n_components;
    if (n_neighbors) *n_neighbors = model->n_neighbors;
    if (MN_ratio) *MN_ratio = model->MN_ratio;
    if (FP_ratio) *FP_ratio = model->FP_ratio;
    if (metric) *metric = static_cast<PacMapMetric>(model->distance_metric);

    return 0;
}

} // extern "C"
```

## 4. Build System Updates

### 4.1 Update CMakeLists.txt

```cmake
# Replace UMAP references with PACMAP
project(PACMAP VERSION 1.0.0)

# Update library name
add_library(pacmap SHARED
    pacmap_simple_wrapper.cpp
    pacmap_fit.cpp
    pacmap_triplet_sampling.cpp
    pacmap_gradient.cpp
    pacmap_optimization.cpp
    # ... other source files
)

# Keep HNSW and other existing dependencies
target_link_libraries(pacmap PRIVATE hnswlib ${CMAKE_THREAD_LIBS_INIT})

# Update output names
set_target_properties(pacmap PROPERTIES
    OUTPUT_NAME "pacmap"
    VERSION ${PROJECT_VERSION}
)
```

## 5. Testing Strategy

### 5.1 Unit Tests

```cpp
// File: test_pacmap_core.cpp

#include "pacmap_simple_wrapper.h"
#include <cassert>
#include <iostream>

void test_triplet_sampling() {
    // Test neighbor pair sampling
    // Test mid-near pair sampling
    // Test far pair sampling
    // Validate triplet distributions
}

void test_weight_schedule() {
    // Test three-phase weight transitions
    // Validate boundary conditions
    // Check smooth interpolation
}

void test_gradient_computation() {
    // Test gradient computation for each triplet type
    // Validate against Python reference
    // Check numerical stability
}

int main() {
    test_triplet_sampling();
    test_weight_schedule();
    test_gradient_computation();

    std::cout << "All PACMAP core tests passed!" << std::endl;
    return 0;
}
```

### 5.2 Integration Tests

```cpp
// File: test_pacmap_integration.cpp

#include "pacmap_simple_wrapper.h"
#include <random>
#include <iostream>

void compare_with_python_reference() {
    // Load test dataset and Python reference results
    // Run PACMAP C++ implementation
    // Compare embeddings within tolerance
    // Validate convergence behavior
}

void test_end_to_end_pipeline() {
    // Test complete fit -> save -> load -> transform pipeline
    // Validate model persistence
    // Test different parameter combinations
}

int main() {
    compare_with_python_reference();
    test_end_to_end_pipeline();

    std::cout << "All PACMAP integration tests passed!" << std::endl;
    return 0;
}
```

## 6. Migration Checklist

### 6.1 Core Implementation
- [ ] Implement triplet sampling functions
- [ ] Implement three-phase weight schedule
- [ ] Implement PACMAP gradient computation
- [ ] Implement main optimization loop
- [ ] Update C++ API wrapper
- [ ] Update build system

### 6.2 Integration with Existing Infrastructure
- [ ] Reuse HNSW optimization for neighbor finding
- [ ] Keep existing distance metric implementations
- [ ] Adapt model persistence for PACMAP structures
- [ ] Keep quantization and compression features
- [ ] Maintain progress reporting infrastructure

### 6.3 Testing and Validation
- [ ] Create unit tests for all new functions
- [ ] Compare results with Python reference implementation
- [ ] Validate performance benchmarks
- [ ] Test cross-platform compatibility

This implementation guide provides the detailed technical specifications needed to implement PACMAP in C++ while leveraging the excellent UMAP infrastructure that already exists.