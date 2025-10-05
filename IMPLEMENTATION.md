# PACMAPCSharp Implementation Guide

## Overview

This document provides detailed step-by-step instructions for implementing PACMAP in C++, based on the Python reference implementation and the existing UMAP infrastructure.

## Current Implementation Status (Updated 2025-10-05)

### Migration Analysis Complete
After comprehensive analysis of the existing codebase, the current implementation status is as follows:

#### File Renaming Completed ✅
- **35 C++ files** have been renamed from `uwot_*` to `pacmap_*`
- All files in `C:\PacMapDotnet\src\pacmap_pure_cpp\` show PACMAP naming
- Directory structure prepared for PACMAP implementation

#### Core Algorithm Status: NOT IMPLEMENTED ❌
- **C# Code**: Still contains UMAP implementation (UMapModel class, UMAP parameters)
- **C++ Code**: Still contains UMAP algorithm despite PACMAP naming
- **Missing PACMAP Features**:
  - Triplet sampling (neighbor, mid-near, far pairs)
  - Three-phase optimization with dynamic weight adjustment
  - PACMAP-specific gradient computation
  - PACMAP parameters (MN_ratio, FP_ratio vs UMAP's minDist, spread)

#### Infrastructure Status: READY ✅
- HNSW optimization infrastructure available and ready for PACMAP
- Distance metrics implementation complete and reusable
- Model persistence infrastructure exists
- Build system prepared for PACMAP compilation
- Cross-platform support maintained

#### Architecture Design: COMPLETED ✅
- **8 PACMAP header files** designed with complete API structure
- **C# API framework** migrated from UMAPuwotSharp to PACMAPuwotSharp
- **Build system updated** for PACMAP structure
- **Testing framework** planned and structured

#### **CRITICAL NEXT STEP: C++ Source Implementation Needed** ⚠️
All header files are designed but **zero C++ implementation files exist**. The next critical phase is implementing the actual PACMAP algorithm in C++.

#### KNN Implementation Issues Identified ⚠️
Analysis of Python reference code shows inconsistencies:
- Some code paths use direct KNN search
- Other code paths use HNSW-enabled PACMAP
- Performance optimization required: **PACMAP KNN must be faster than UMAP, not slower**

### Implementation Priority Matrix (Updated After Review)

| Component | Status | Priority | Complexity | Timeline | Review Findings |
|-----------|--------|----------|------------|----------|-----------------|
| **C++ Source Implementation** | ❌ CRITICAL | CRITICAL | High | 1-2 weeks | Headers designed, zero .cpp files implemented |
| **C# API Migration** | ❌ CRITICAL | CRITICAL | High | 2-3 days | Still UMapModel class, namespace needs PACMAP |
| **Adam Optimizer** | ❌ Missing | HIGH | Medium | 2 days | Review recommends Adam over GD for stability |
| **Triplet Sampling** | ❌ Missing | HIGH | Medium | 2-3 days | Review confirms HNSW optimization approach |
| **Three-Phase Weights** | ❌ Missing | HIGH | Low | 1 day | Aligns with review specification |
| **Error Handling** | ❌ Missing | HIGH | Medium | 1-2 days | Review emphasizes robust error codes |
| **Gradient Computation** | ❌ Missing | HIGH | Medium | 2 days | Review provides optimized parallel implementation |
| **KNN Optimization** | ⚠️ Issues | MEDIUM | High | 3-4 days | Review addresses performance bottlenecks |
| **Cross-Platform Determinism** | ❌ Missing | MEDIUM | Low | 1 day | Review identifies FP precision issues |
| **Testing/Validation** | ⚠️ Partial | MEDIUM | Medium | 2-3 days | Review suggests comprehensive risk testing |

**TOTAL ESTIMATED TIME: 2-3 weeks for functional PACMAP implementation**

## Comprehensive Implementation Plan (Review-Optimized)

### Complete Implementation Gap Analysis

#### **Critical Gaps Identified**

1. **C# API Still UMAP-Based** - CRITICAL GAP
   - Current: `UMapModel` class, `UMAPuwotSharp` namespace
   - Required: `PacMapModel` class, `PACMAPuwotSharp` namespace
   - Impact: Entire C# public API needs migration

2. **Missing Adam Optimizer** - CRITICAL GAP
   - Current: Simple gradient descent planned
   - Required: Adam optimizer with β₁=0.9, β₂=0.999, ε=1e-8
   - Impact: Convergence stability and speed

3. **Incomplete Triplet Sampling** - CRITICAL GAP
   - Current: Basic HNSW neighbor sampling
   - Required: Three-type triplet sampling with distance-based filtering
   - Impact: Algorithm correctness and performance

4. **Missing Error Framework** - HIGH GAP
   - Current: Basic error handling
   - Required: UWOT_ERROR_* style codes, comprehensive edge case handling
   - Impact: Production readiness and user experience

### Complete File Structure Analysis

#### **NEW FILES TO IMPLEMENT**

```cpp
// ==== CORE ALGORITHM FILES (NEW) ====

// File: pacmap_triplet_sampling.h
// Purpose: HNSW-optimized triplet sampling for all three pair types
// Dependencies: hnswlib, pacmap_model.h, pacmap_distance.h
// Key Functions:
//   - void sample_triplets(PacMapModel* model, float* data, uwot_progress_callback_v2 callback)
//   - void sample_neighbors_pair(PacMapModel* model, std::vector<Triplet>& neighbor_triplets)
//   - void sample_MN_pair(PacMapModel* model, std::vector<Triplet>& mn_triplets)
//   - void sample_FP_pair(PacMapModel* model, std::vector<Triplet>& fp_triplets)
//   - std::mt19937 get_seeded_rng(int seed)

// File: pacmap_triplet_sampling.cpp
// Implementation: Parallel triplet sampling with HNSW optimization
// Features: OpenMP parallelization, distance-based filtering, oversampling handling

// File: pacmap_gradient.h
// Purpose: Adam-optimized gradient computation with parallel processing
// Dependencies: pacmap_model.h, <omp.h>
// Key Functions:
//   - std::tuple<float, float, float> get_weights(int current_iter, int total_iters)
//   - void compute_gradients(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
//                           std::vector<float>& gradients, float w_n, float w_mn, float w_f, int n_components)
//   - void adam_update(std::vector<float>& embedding, const std::vector<float>& gradients,
//                      std::vector<float>& m, std::vector<float>& v, int iter, float learning_rate,
//                      float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)

// File: pacmap_gradient.cpp
// Implementation: Parallel gradient computation with Adam optimizer
// Features: OpenMP atomic operations, numerical stability, efficient memory access

// File: pacmap_optimization.h
// Purpose: Three-phase optimization loop with Adam optimizer
// Dependencies: pacmap_model.h, pacmap_gradient.h, pacmap_triplet_sampling.h
// Key Functions:
//   - void optimize_embedding(PacMapModel* model, float* embedding_out, uwot_progress_callback_v2 callback)
//   - void initialize_random_embedding(std::vector<float>& embedding, int n_samples, int n_components, std::mt19937& rng)
//   - void compute_safety_stats(PacMapModel* model, const std::vector<float>& embedding)

// File: pacmap_optimization.cpp
// Implementation: Three-phase optimization with Adam and progress reporting
// Features: Adaptive phase transitions, loss monitoring, convergence detection

// File: pacmap_transform.h
// Purpose: Weighted nearest neighbor interpolation for new data points
// Dependencies: pacmap_model.h, hnswlib
// Key Functions:
//   - int pacmap_transform_detailed(PacMapModel* model, float* new_data, int n_new_obs, int n_dim,
//                                   float* embedding, int* nn_indices, float* nn_distances,
//                                   float* confidence_score, int* outlier_level)
//   - void compute_confidence_scores(PacMapModel* model, const std::vector<float>& new_embedding,
//                                    float* confidence_scores, int* outlier_levels)

// File: pacmap_transform.cpp
// Implementation: Weighted NN interpolation with safety analysis
// Features: Outlier detection, confidence scoring, parallel processing

// File: pacmap_persistence.h
// Purpose: Enhanced model save/load with CRC32 validation
// Dependencies: pacmap_model.h, <fstream>, <crc32>
// Key Functions:
//   - int pacmap_save_model(PacMapModel* model, const char* filename)
//   - PacMapModel* pacmap_load_model(const char* filename)
//   - uint32_t compute_crc32(const void* data, size_t size)
//   - void serialize_triplets(std::ofstream& file, const std::vector<Triplet>& triplets)
//   - void deserialize_triplets(std::ifstream& file, std::vector<Triplet>& triplets)

// File: pacmap_persistence.cpp
// Implementation: Binary serialization with CRC validation
// Features: Version compatibility, data integrity checks, error recovery

// ==== UTILITY FILES (NEW) ====

// File: pacmap_utils.h
// Purpose: Error handling, validation, and utility functions
// Dependencies: pacmap_model.h
// Key Functions:
//   - int validate_parameters(PacMapModel* model)
//   - void validate_metric_data(const float* data, int n_obs, int n_dim, PacMapMetric metric)
//   - const char* pacmap_get_error_message(int error_code)
//   - bool check_memory_requirements(int n_samples, int n_features, int n_neighbors)
//   - void auto_tune_parameters(PacMapModel* model, int n_samples)

// File: pacmap_utils.cpp
// Implementation: Parameter validation, error handling, auto-tuning
// Features: Comprehensive edge case detection, helpful error messages

// File: pacmap_determinism.h
// Purpose: Cross-platform determinism controls
// Dependencies: pacmap_model.h
// Key Functions:
//   - void enable_strict_fp_mode(bool enable)
//   - void set_deterministic_thread_count(int count)
//   - void validate_determinism_settings()

// File: pacmap_determinism.cpp
// Implementation: Platform-specific determinism controls
// Features: FP consistency, thread pool management, validation

// ==== UPDATED FILES (MODIFY EXISTING) ====

// File: pacmap_model.h (Updated from UMAP)
// Purpose: Core PACMAP data structures
// Changes: Replace UMAP structures with PACMAP equivalents

// File: pacmap_simple_wrapper.h (Updated from UMAP)
// Purpose: C API wrapper for C# integration
// Changes: Update function signatures for PACMAP parameters

// File: pacmap_simple_wrapper.cpp (Updated from UMAP)
// Purpose: C API implementation
// Changes: Replace UMAP function calls with PACMAP equivalents

// File: CMakeLists.txt (Updated)
// Purpose: Build configuration
// Changes: Add new source files, update compiler flags for determinism
```

#### **OBSOLETE FILES TO REMOVE**

```cpp
// ==== UMAP-SPECIFIC FILES (OBSOLETE) ====

// File: uwot_simple_wrapper.h -> REMOVE
// Reason: Replaced by pacmap_simple_wrapper.h
// Migration: Copy function signatures, update parameter names

// File: uwot_simple_wrapper.cpp -> REMOVE
// Reason: Replaced by pacmap_simple_wrapper.cpp
// Migration: Update function implementations for PACMAP

// File: uwot_optimization.h -> REMOVE
// Reason: Replaced by pacmap_optimization.h
// Migration: Copy optimization patterns, update for Adam

// File: uwot_optimization.cpp -> REMOVE
// Reason: Replaced by pacmap_optimization.cpp
// Migration: Update gradient descent to Adam optimizer

// File: uwot_transform.h -> REMOVE
// Reason: Replaced by pacmap_transform.h
// Migration: Copy transform logic, update for PACMAP triplets

// File: uwot_transform.cpp -> REMOVE
// Reason: Replaced by pacmap_transform.cpp
// Migration: Update transform implementation for PACMAP

// File: uwot_grad.h -> REMOVE
// Reason: Replaced by pacmap_gradient.h
// Migration: Copy gradient patterns, update for triplet-based computation

// File: uwot_grad.cpp -> REMOVE
// Reason: Replaced by pacmap_gradient.cpp
// Migration: Update gradient computation for three triplet types

// File: uwot_graph.cpp -> REMOVE
// Reason: PACMAP uses triplets, not fuzzy simplicial sets
// Migration: No direct migration needed

// File: uwot_simplicial_set.cpp -> REMOVE
// Reason: PACMAP doesn't use simplicial sets
// Migration: No direct migration needed

// ==== UMAP TEST FILES (OBSOLETE) ====

// File: tests/*UMap*.cs -> REMOVE/UPDATE
// Reason: Need PACMAP-specific tests
// Migration: Update test cases for PACMAP behavior
```

### Implementation Priority Matrix (Comprehensive)

| Component | Status | Priority | Complexity | Files Involved | Dependencies | Timeline |
|-----------|--------|----------|------------|----------------|--------------|----------|
| **C# API Migration** | ❌ CRITICAL | CRITICAL | High | PacMapModel.cs, *.Tests.cs | C++ completion | 3-4 days |
| **Core Data Structures** | ❌ Missing | CRITICAL | Medium | pacmap_model.h/cpp | None | 1-2 days |
| **Triplet Sampling** | ❌ Missing | HIGH | High | pacmap_triplet_sampling.h/cpp | HNSW, distance metrics | 3-4 days |
| **Adam Gradient System** | ❌ Missing | HIGH | High | pacmap_gradient.h/cpp | Data structures | 2-3 days |
| **Three-Phase Optimization** | ❌ Missing | HIGH | Medium | pacmap_optimization.h/cpp | Gradient system | 2-3 days |
| **Transform System** | ❌ Missing | MEDIUM | Medium | pacmap_transform.h/cpp | Core algorithm | 2 days |
| **Persistence System** | ❌ Missing | MEDIUM | Medium | pacmap_persistence.h/cpp | Data structures | 2 days |
| **Error Framework** | ❌ Missing | HIGH | Medium | pacmap_utils.h/cpp | All components | 1-2 days |
| **Determinism Controls** | ❌ Missing | MEDIUM | Low | pacmap_determinism.h/cpp | None | 1 day |
| **C# Wrapper Update** | ❌ Missing | HIGH | Low | pacmap_simple_wrapper.h/cpp | C++ completion | 1 day |
| **Build System Update** | ❌ Missing | MEDIUM | Low | CMakeLists.txt | All files | 0.5 day |
| **Test Suite Creation** | ❌ Missing | MEDIUM | Medium | Multiple test files | All components | 3-4 days |

**Total Estimated Timeline: 21-30 days (3-4 weeks)**

#### 2.6 Complete Error Handling Framework (Review-Optimized)

```cpp
// File: pacmap_utils.h (Complete Error Handling)

#pragma once

#include "pacmap_model.h"
#include <string>
#include <vector>

// Error codes (following UMAP pattern)
enum PacMapErrorCode {
    PACMAP_SUCCESS = 0,
    PACMAP_ERROR_INVALID_PARAMETERS = 1,
    PACMAP_ERROR_INSUFFICIENT_MEMORY = 2,
    PACMAP_ERROR_INVALID_DATA = 3,
    PACMAP_ERROR_DIMENSION_MISMATCH = 4,
    PACMAP_ERROR_INVALID_METRIC = 5,
    PACMAP_ERROR_HNSW_FAILURE = 6,
    PACMAP_ERROR_FILE_IO = 7,
    PACMAP_ERROR_CRC_MISMATCH = 8,
    PACMAP_ERROR_MODEL_NOT_FITTED = 9,
    PACMAP_ERROR_QUANTIZATION_FAILURE = 10,
    PACMAP_ERROR_OPTIMIZATION_FAILURE = 11,
    PACMAP_ERROR_UNKNOWN = 999
};

// Parameter validation functions
int validate_parameters(PacMapModel* model);
void validate_metric_data(const float* data, int n_obs, int n_dim, PacMapMetric metric);
bool check_memory_requirements(int n_samples, int n_features, int n_neighbors);
void auto_tune_parameters(PacMapModel* model, int n_samples);

// Error handling utilities
const char* pacmap_get_error_message(int error_code);
void set_last_error(PacMapModel* model, int error_code, const std::string& message);
int get_last_error_code(PacMapModel* model);
const char* get_last_error_message(PacMapModel* model);

// Edge case detection
bool detect_degenerate_cases(int n_samples, int n_features);
bool check_for_nan_inf(const float* data, int size);
bool validate_triplet_distribution(const std::vector<Triplet>& triplets, int n_samples);
```

```cpp
// File: pacmap_utils.cpp (Complete Error Handling Implementation)

#include "pacmap_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

int validate_parameters(PacMapModel* model) {
    if (!model) return PACMAP_ERROR_INVALID_PARAMETERS;

    // Validate basic parameters
    if (model->n_samples <= 0 || model->n_features <= 0) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMETERS,
                      "n_samples and n_features must be positive");
        return PACMAP_ERROR_INVALID_PARAMETERS;
    }

    if (model->n_neighbors <= 0 || model->n_neighbors >= model->n_samples) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMETERS,
                      "n_neighbors must be between 1 and n_samples-1");
        return PACMAP_ERROR_INVALID_PARAMETERS;
    }

    // Validate PACMAP-specific parameters
    if (model->mn_ratio < 0.0f || model->fp_ratio < 0.0f) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMETERS,
                      "MN_ratio and FP_ratio must be non-negative");
        return PACMAP_ERROR_INVALID_PARAMETERS;
    }

    if (model->learning_rate <= 0.0f) {
        set_last_error(model, PACMAP_ERROR_INVALID_PARAMETERS,
                      "learning_rate must be positive");
        return PACMAP_ERROR_INVALID_PARAMETERS;
    }

    return PACMAP_SUCCESS;
}

const char* pacmap_get_error_message(int error_code) {
    switch (error_code) {
        case PACMAP_SUCCESS: return "Success";
        case PACMAP_ERROR_INVALID_PARAMETERS: return "Invalid parameters provided";
        case PACMAP_ERROR_INSUFFICIENT_MEMORY: return "Insufficient memory for operation";
        case PACMAP_ERROR_INVALID_DATA: return "Invalid data (NaN, Inf, or format issues)";
        case PACMAP_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch between inputs";
        case PACMAP_ERROR_INVALID_METRIC: return "Invalid distance metric for data type";
        case PACMAP_ERROR_HNSW_FAILURE: return "HNSW index construction or search failed";
        case PACMAP_ERROR_FILE_IO: return "File I/O operation failed";
        case PACMAP_ERROR_CRC_MISMATCH: return "CRC32 validation failed - file corruption";
        case PACMAP_ERROR_MODEL_NOT_FITTED: return "Model has not been fitted - call FitTransform first";
        default: return "Unknown error";
    }
}
```

#### 2.7 Complete C# API Migration (Review-Optimized)

```csharp
// File: PacMapModel.cs (Complete PACMAP Implementation - Updated from UMAP)

using System;
using System.IO;
using System.Runtime.InteropServices;

namespace PACMAPuwotSharp  // Updated namespace from UMAPuwotSharp
{
    /// <summary>
    /// PACMAP (Pairwise Controlled Manifold Approximation and Projection) implementation
    /// High-performance dimensionality reduction using triplet-based optimization
    /// </summary>
    public class PacMapModel : IDisposable  // Updated from UMapModel
    {
        private IntPtr _modelHandle;
        private bool _disposed = false;

        // PACMAP-specific parameters (NEW - replacing UMAP parameters)
        public float MN_ratio { get; private set; } = 0.5f;
        public float FP_ratio { get; private set; } = 2.0f;
        public float LearningRate { get; private set; } = 1.0f;
        public (int phase1, int phase2, int phase3) NumIters { get; private set; } = (100, 100, 250);

        // Common parameters (kept from UMAP)
        public int NComponents { get; private set; } = 2;
        public int NNeighbors { get; private set; } = 10;
        public DistanceMetric Distance { get; private set; } = DistanceMetric.Euclidean;
        public int RandomSeed { get; private set; } = -1;
        public bool UseQuantization { get; private set; } = false;

        /// <summary>
        /// Creates a new PACMAP model with specified parameters
        /// </summary>
        public PacMapModel(
            int n_components = 2,
            int n_neighbors = 10,
            float mn_ratio = 0.5f,      // NEW: PACMAP-specific
            float fp_ratio = 2.0f,      // NEW: PACMAP-specific
            float learning_rate = 1.0f,  // NEW: PACMAP-specific
            (int, int, int) num_iters = (100, 100, 250),  // NEW: PACMAP three-phase
            DistanceMetric distance = DistanceMetric.Euclidean,
            bool force_exact_knn = false,
            bool use_quantization = false,
            int random_seed = -1)
        {
            // Set PACMAP-specific parameters
            NComponents = n_components;
            NNeighbors = n_neighbors;
            MN_ratio = mn_ratio;
            FP_ratio = fp_ratio;
            LearningRate = learning_rate;
            NumIters = num_iters;
            Distance = distance;
            UseQuantization = use_quantization;
            RandomSeed = random_seed;

            // Create native model
            _modelHandle = NativeMethods.pacmap_create_model();
            if (_modelHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to create PACMAP model");
            }
        }

        /// <summary>
        /// Fits the model to data and returns the embedded coordinates
        /// </summary>
        public float[,] FitTransform(float[,] data,
            int embedding_dimension = 2,
            int n_neighbors = 10,
            float mn_ratio = 0.5f,      // PACMAP-specific
            float fp_ratio = 2.0f,      // PACMAP-specific
            float learning_rate = 1.0f,  // PACMAP-specific
            (int, int, int) num_iters = (100, 100, 250),  // PACMAP three-phase
            DistanceMetric distance = DistanceMetric.Euclidean,
            bool force_exact_knn = false,
            bool use_quantization = false,
            int random_seed = -1)
        {
            ValidateInputData(data);

            int n_obs = data.GetLength(0);
            int n_dim = data.GetLength(1);
            float[,] embedding = new float[n_obs, embedding_dimension];

            int result = NativeMethods.pacmap_fit_with_progress_v2(
                _modelHandle, data, n_obs, n_dim, embedding_dimension,
                n_neighbors, mn_ratio, fp_ratio,
                num_iters.Item1, num_iters.Item2, num_iters.Item3,
                (int)distance, embedding, null,
                force_exact_knn ? 1 : 0, 16, 200, 200,
                use_quantization ? 1 : 0, random_seed, 0);

            if (result != 0)
            {
                throw new InvalidOperationException($"PACMAP fitting failed with error code: {result}");
            }

            return embedding;
        }

        /// <summary>
        /// Transforms new data points using the fitted model
        /// </summary>
        public float[,] Transform(float[,] data)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(PacMapModel));
            ValidateInputData(data);

            int n_new = data.GetLength(0);
            float[,] embedding = new float[n_new, NComponents];

            int result = NativeMethods.pacmap_transform_detailed(
                _modelHandle, data, n_new, data.GetLength(1), embedding,
                null, null, null, null);

            if (result != 0)
            {
                throw new InvalidOperationException($"PACMAP transform failed with error code: {result}");
            }

            return embedding;
        }

        /// <summary>
        /// Saves the trained model to a file
        /// </summary>
        public void SaveModel(string filename)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(PacMapModel));
            int result = NativeMethods.pacmap_save_model(_modelHandle, filename);
            if (result != 0) throw new InvalidOperationException($"Failed to save model: {filename}");
        }

        /// <summary>
        /// Loads a trained model from a file
        /// </summary>
        public static PacMapModel Load(string filename)
        {
            IntPtr modelHandle = NativeMethods.pacmap_load_model(filename);
            if (modelHandle == IntPtr.Zero) throw new InvalidOperationException($"Failed to load model: {filename}");
            return new PacMapModel(modelHandle);
        }

        private void ValidateInputData(float[,] data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (data.GetLength(0) <= 0 || data.GetLength(1) <= 0)
                throw new ArgumentException("Data must have positive dimensions");

            // Check for NaN/Inf values
            for (int i = 0; i < data.GetLength(0); i++)
                for (int j = 0; j < data.GetLength(1); j++)
                    if (!float.IsFinite(data[i, j]))
                        throw new ArgumentException($"Data contains NaN or Inf values at [{i}, {j}]");
        }

        private PacMapModel(IntPtr modelHandle) { _modelHandle = modelHandle; }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _modelHandle != IntPtr.Zero)
            {
                NativeMethods.pacmap_destroy_model(_modelHandle);
                _modelHandle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~PacMapModel() { Dispose(false); }
    }

    /// <summary>
    /// Native method declarations for PACMAP C++ library
    /// </summary>
    internal static class NativeMethods
    {
        private const string DllName = "pacmap";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pacmap_create_model();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pacmap_destroy_model(IntPtr model);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pacmap_fit_with_progress_v2(
            IntPtr model, float[,] data, int n_obs, int n_dim, int embedding_dim,
            int n_neighbors, float mn_ratio, float fp_ratio,
            int phase1_iters, int phase2_iters, int phase3_iters,
            int metric, float[,] embedding, IntPtr callback,
            int force_exact_knn, int M, int ef_construction, int ef_search,
            int use_quantization, int random_seed, int autoHNSWParam);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pacmap_transform_detailed(
            IntPtr model, float[,] new_data, int n_new_obs, int n_dim,
            float[,] embedding, int[] nn_indices, float[] nn_distances,
            float[] confidence_score, int[] outlier_level);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pacmap_save_model(IntPtr model, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pacmap_load_model([MarshalAs(UnmanagedType.LPStr)] string filename);
    }
}
```

## Complete Migration Checklist (Review-Optimized)

### **Phase 1: Critical Infrastructure (Week 1)**
- [ ] **CRITICAL**: Migrate C# API from UMapModel to PacMapModel
- [ ] **CRITICAL**: Update namespace from UMAPuwotSharp to PACMAPuwotSharp
- [ ] **CRITICAL**: Replace UMAP parameters (minDist, spread) with PACMAP parameters (MN_ratio, FP_ratio, lr, num_iters)
- [ ] Implement core PACMAP data structures (pacmap_model.h/cpp)
- [ ] Create comprehensive error handling framework (pacmap_utils.h/cpp)

### **Phase 2: Core Algorithm Implementation (Week 2)**
- [ ] Implement HNSW-optimized triplet sampling system (pacmap_triplet_sampling.h/cpp)
- [ ] Implement Adam optimizer with parallel gradients (pacmap_gradient.h/cpp)
- [ ] Implement three-phase optimization loop (pacmap_optimization.h/cpp)
- [ ] Add cross-platform determinism controls (pacmap_determinism.h/cpp)

### **Phase 3: Integration and Features (Week 3)**
- [ ] Implement weighted NN transform system (pacmap_transform.h/cpp)
- [ ] Implement enhanced persistence with CRC validation (pacmap_persistence.h/cpp)
- [ ] Update C++ wrapper for PACMAP API (pacmap_simple_wrapper.h/cpp)
- [ ] Update build system with determinism flags (CMakeLists.txt)

### **Phase 4: Testing and Validation (Week 4)**
- [ ] Create comprehensive test suite covering all components
- [ ] Validate embeddings against Python reference implementation
- [ ] Benchmark performance vs UMAP (must be faster for KNN)
- [ ] Test cross-platform determinism and consistency

### **File Cleanup (Post-Migration)**
- [ ] Remove all UMAP-specific files (uwot_*.h/cpp)
- [ ] Remove or update UMAP-specific test files
- [ ] Update documentation and examples for PACMAP API
- [ ] Clean up any remaining UMAP references in codebase

### **Success Criteria**
1. **Functional Correctness**: Embeddings match Python reference within 1% tolerance
2. **Performance**: PACMAP KNN faster than UMAP, overall performance within 10% of UMAP
3. **API Compatibility**: Clean migration from UMAP to PACMAP with minimal user code changes
4. **Production Ready**: Comprehensive error handling, parameter validation, safety features
5. **Cross-Platform**: Identical results across Windows and Linux with fixed seeds

This comprehensive implementation plan provides everything needed to successfully migrate from UMAP to PACMAP while leveraging 95% of existing infrastructure and incorporating all review-identified improvements for production-ready performance and robustness.

## C++ Implementation Details (Updated for Review)

### 1. Data Structures

#### 1.1 PACMAP Model Structure (Review-Optimized)

```cpp
// File: pacmap_model.h (Review-Optimized)

#pragma once
#include <vector>
#include <memory>
#include <random>
#include <hnswlib/hnswlib.h>

// Distance metrics (enhanced from UMAP)
enum PacMapMetric {
    PACMAP_METRIC_EUCLIDEAN = 0,
    PACMAP_METRIC_COSINE = 1,
    PACMAP_METRIC_MANHATTAN = 2,
    PACMAP_METRIC_CORRELATION = 3,
    PACMAP_METRIC_HAMMING = 4
};

// Triplet types (review-optimized)
enum TripletType {
    NEIGHBOR = 0,
    MID_NEAR = 1,
    FURTHER = 2
};

// Unified triplet structure (review recommendation)
struct Triplet {
    int anchor, neighbor;
    TripletType type;
    float weight = 1.0f;
};

// Enhanced PACMAP Model Structure (review-optimized)
struct PacMapModel {
    // Core parameters
    int n_samples = 0, n_features = 0, n_components = 2, n_neighbors = 10;
    float mn_ratio = 0.5f, fp_ratio = 2.0f, learning_rate = 1.0f;
    int phase1_iters = 100, phase2_iters = 100, phase3_iters = 250;
    PacMapMetric metric = PACMAP_METRIC_EUCLIDEAN;
    int random_seed = -1;  // -1: non-deterministic, else seeded

    // HNSW parameters (from UMAP)
    int hnsw_m = 16, hnsw_ef_construction = 200, hnsw_ef_search = 200;
    bool use_quantization = false;

    // Unified triplet storage (review optimization)
    std::vector<Triplet> triplets;

    // Data storage
    std::vector<float> training_data, embedding;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> original_space_index;

    // Safety statistics (review addition)
    float min_embedding_dist = 0, p95_embedding_dist = 0;

    // RNG for deterministic behavior (review requirement)
    std::mt19937 rng;

    // Error handling (review addition)
    int last_error_code = 0;
    std::string last_error_message;
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

#### 2.3 Complete Triplet Sampling Implementation (Review-Optimized)

```cpp
// File: pacmap_triplet_sampling.h

#pragma once

#include "pacmap_model.h"
#include <vector>
#include <random>
#include <unordered_set>

// Utility functions for RNG management
std::mt19937 get_seeded_rng(int seed);

// Core triplet sampling functions
void sample_triplets(PacMapModel* model, float* data, uwot_progress_callback_v2 callback);
void sample_neighbors_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                         std::vector<Triplet>& neighbor_triplets);
void sample_MN_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& mn_triplets, int n_mn);
void sample_FP_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& fp_triplets, int n_fp);

// Distance-based sampling helpers
void distance_based_sampling(PacMapModel* model, const std::vector<float>& data,
                           int target_pairs, float min_dist, float max_dist,
                           std::vector<Triplet>& triplets, TripletType type);
std::vector<float> compute_distance_percentiles(const std::vector<float>& data, int n_samples, int n_features);
```

```cpp
// File: pacmap_triplet_sampling.cpp (Complete Implementation)

#include "pacmap_triplet_sampling.h"
#include "pacmap_distance.h"
#include "pacmap_utils.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

std::mt19937 get_seeded_rng(int seed) {
    return seed >= 0 ? std::mt19937(seed) : std::mt19937(std::random_device{}());
}

void sample_triplets(PacMapModel* model, float* data, uwot_progress_callback_v2 callback) {
    // Validate parameters first
    if (validate_parameters(model) != PACMAP_SUCCESS) {
        return;
    }

    // Normalize/quantize data (reuse UMAP infrastructure)
    std::vector<float> normalized_data;
    normalized_data.assign(data, data + model->n_samples * model->n_features);

    if (model->use_quantization) {
        // Apply quantization (reuse from UMAP)
        // quantize_data(normalized_data, model->n_features);
    }

    // Save normalized data for transform
    model->training_data = normalized_data;

    // Build HNSW index for efficient neighbor search
    model->original_space_index = create_hnsw_index(normalized_data.data(),
                                                    model->n_samples,
                                                    model->n_features,
                                                    model->metric,
                                                    model->hnsw_m,
                                                    model->hnsw_ef_construction);

    // Initialize RNG for deterministic behavior
    model->rng = get_seeded_rng(model->random_seed);

    // Sample three types of triplets
    std::vector<Triplet> neighbor_triplets, mn_triplets, fp_triplets;

    // Neighbor pairs using HNSW
    sample_neighbors_pair(model, normalized_data, neighbor_triplets);

    // Mid-near pairs with distance-based sampling
    int n_mn = static_cast<int>(model->n_neighbors * model->mn_ratio);
    sample_MN_pair(model, normalized_data, mn_triplets, n_mn);

    // Far pairs for uniform distribution
    int n_fp = static_cast<int>(model->n_neighbors * model->fp_ratio);
    sample_FP_pair(model, normalized_data, fp_triplets, n_fp);

    // Combine all triplets (review-optimized: single vector)
    model->triplets.clear();
    model->triplets.reserve(neighbor_triplets.size() + mn_triplets.size() + fp_triplets.size());
    model->triplets.insert(model->triplets.end(), neighbor_triplets.begin(), neighbor_triplets.end());
    model->triplets.insert(model->triplets.end(), mn_triplets.begin(), mn_triplets.end());
    model->triplets.insert(model->triplets.end(), fp_triplets.begin(), fp_triplets.end());

    callback("Sampling Triplets", 100, 100.0f, nullptr);
}

void sample_neighbors_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                         std::vector<Triplet>& neighbor_triplets) {

    neighbor_triplets.clear();
    neighbor_triplets.reserve(model->n_samples * model->n_neighbors);

    // Parallel neighbor sampling using HNSW (review optimization)
    #pragma omp parallel for
    for (int i = 0; i < model->n_samples; ++i) {
        auto knn = model->original_space_index->searchKnn(
            normalized_data.data() + i * model->n_features,
            model->n_neighbors + 1);

        std::vector<Triplet> local_triplets;
        for (size_t j = 1; j < knn.size(); ++j) {  // Skip self (first result)
            local_triplets.emplace_back(Triplet{i, static_cast<int>(knn[j].second), NEIGHBOR});
        }

        // Merge results safely
        #pragma omp critical
        {
            neighbor_triplets.insert(neighbor_triplets.end(),
                                   local_triplets.begin(), local_triplets.end());
        }
    }
}

void sample_MN_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& mn_triplets, int n_mn) {

    mn_triplets.clear();

    // Compute distance percentiles for mid-near range (25th-75th percentile)
    auto percentiles = compute_distance_percentiles(normalized_data,
                                                   std::min(model->n_samples, 1000),
                                                   model->n_features);
    float p25_dist = percentiles[0];
    float p75_dist = percentiles[1];

    // Distance-based sampling for mid-near pairs
    distance_based_sampling(model, normalized_data,
                           model->n_samples * n_mn * 2,  // Oversample for uniqueness
                           p25_dist, p75_dist,
                           mn_triplets, MID_NEAR);
}

void sample_FP_pair(PacMapModel* model, const std::vector<float>& normalized_data,
                   std::vector<Triplet>& fp_triplets, int n_fp) {

    fp_triplets.clear();

    // Compute 90th percentile for far pairs
    auto percentiles = compute_distance_percentiles(normalized_data,
                                                   std::min(model->n_samples, 500),
                                                   model->n_features);
    float p90_dist = percentiles[2];  // 90th percentile

    // Distance-based sampling for far pairs
    distance_based_sampling(model, normalized_data,
                           model->n_samples * n_fp * 3,  // Oversample more for far pairs
                           p90_dist, std::numeric_limits<float>::infinity(),
                           fp_triplets, FURTHER);
}

void distance_based_sampling(PacMapModel* model, const std::vector<float>& data,
                           int target_pairs, float min_dist, float max_dist,
                           std::vector<Triplet>& triplets, TripletType type) {

    std::uniform_int_distribution<int> dist(0, model->n_samples - 1);
    std::unordered_set<long long> used_pairs;
    int pairs_found = 0;

    // Adaptive sampling loop with oversampling
    int max_attempts = target_pairs * 10;  // Prevent infinite loops
    int attempts = 0;

    while (pairs_found < target_pairs && attempts < max_attempts) {
        int i = dist(model->rng);
        int j = dist(model->rng);

        if (i == j) continue;

        // Ensure unique pairs using bit packing
        long long pair_key = ((long long)std::min(i, j) << 32) | std::max(i, j);
        if (used_pairs.find(pair_key) != used_pairs.end()) continue;

        // Compute distance with early termination for efficiency
        float distance = compute_distance(data.data() + i * model->n_features,
                                        data.data() + j * model->n_features,
                                        model->n_features, model->metric);

        if (distance >= min_dist && distance <= max_dist) {
            triplets.emplace_back(Triplet{i, j, type});
            used_pairs.insert(pair_key);
            pairs_found++;
        }

        attempts++;
    }
}

std::vector<float> compute_distance_percentiles(const std::vector<float>& data, int n_samples, int n_features) {
    std::vector<float> distances;

    // Sample distances for percentile estimation (optimize for large datasets)
    int sample_size = std::min(n_samples, 1000);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = i + 1; j < sample_size; ++j) {
            float dist = compute_distance(data.data() + i * n_features,
                                        data.data() + j * n_features,
                                        n_features, PACMAP_METRIC_EUCLIDEAN);
            distances.push_back(dist);
        }
    }

    std::sort(distances.begin(), distances.end());
    std::vector<float> percentiles(3);

    // 25th, 75th, and 90th percentiles
    percentiles[0] = distances[distances.size() * 0.25];
    percentiles[1] = distances[distances.size() * 0.75];
    percentiles[2] = distances[distances.size() * 0.90];

    return percentiles;
}
```

#### 2.4 Complete Adam Gradient System (Review-Optimized)

```cpp
// File: pacmap_gradient.h (Complete)

#pragma once

#include "pacmap_model.h"
#include <vector>
#include <tuple>

// Three-phase weight schedule
std::tuple<float, float, float> get_weights(int current_iter, int total_iters);

// Parallel gradient computation with atomic operations
void compute_gradients(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                       std::vector<float>& gradients, float w_n, float w_mn, float w_f, int n_components);

// Adam optimizer with bias correction and adaptive learning rates
void adam_update(std::vector<float>& embedding, const std::vector<float>& gradients,
                 std::vector<float>& m, std::vector<float>& v, int iter, float learning_rate,
                 float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

// Loss computation for monitoring convergence
float compute_pacmap_loss(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components);

// Convergence detection
bool check_convergence(const std::vector<float>& loss_history, float threshold = 1e-6, int window = 50);
```

```cpp
// File: pacmap_gradient.cpp (Complete Implementation)

#include "pacmap_gradient.h"
#include "pacmap_distance.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>

std::tuple<float, float, float> get_weights(int current_iter, int total_iters) {
    float progress = static_cast<float>(current_iter) / total_iters;
    float w_n = 1.0f, w_f = 1.0f, w_mn;

    // Three-phase weight schedule (review specification)
    if (progress < 0.1f) {
        // Phase 1: Global structure focus (0-10%)
        // w_MN decreases linearly from 1000 to 3
        float phase_progress = progress * 10.0f;  // 0 to 1 within phase
        w_mn = 1000.0f * (1.0f - phase_progress) + 3.0f * phase_progress;
    } else if (progress < 0.4f) {
        // Phase 2: Balance phase (10-40%)
        w_mn = 3.0f;
    } else {
        // Phase 3: Local structure focus (40-100%)
        // w_MN decreases linearly from 3 to 0
        float phase_progress = (progress - 0.4f) / 0.6f;  // 0 to 1 within phase
        w_mn = 3.0f * (1.0f - phase_progress);
    }

    return {w_n, w_mn, w_f};
}

void compute_gradients(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                       std::vector<float>& gradients, float w_n, float w_mn, float w_f, int n_components) {

    gradients.assign(embedding.size(), 0.0f);

    // Parallel gradient computation with atomic operations (review requirement)
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t idx = 0; idx < triplets.size(); ++idx) {
        const auto& t = triplets[idx];
        size_t idx_a = static_cast<size_t>(t.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(t.neighbor) * n_components;

        // Compute Euclidean distance in embedding space
        float d_ij_squared = 0.0f;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            d_ij_squared += diff * diff;
        }
        float d_ij = std::sqrt(std::max(d_ij_squared, 1e-8f));

        // Compute gradient magnitude based on triplet type (PACMAP loss functions)
        float grad_magnitude;
        switch (t.type) {
            case NEIGHBOR:
                // Attractive force: w * 10 / ((10 + d)^2)
                grad_magnitude = w_n * 10.0f / std::pow(10.0f + d_ij, 2.0f);
                break;
            case MID_NEAR:
                // Moderate attractive force: w * 10000 / ((10000 + d)^2)
                grad_magnitude = w_mn * 10000.0f / std::pow(10000.0f + d_ij, 2.0f);
                break;
            case FURTHER:
                // Repulsive force: -w / ((1 + d)^2)
                grad_magnitude = -w_f / std::pow(1.0f + d_ij, 2.0f);
                break;
            default:
                continue;  // Should never happen
        }

        // Apply gradients symmetrically (Newton's third law)
        float scale = grad_magnitude / d_ij;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            float gradient_component = scale * diff;

            // Thread-safe atomic operations (review requirement for determinism)
            #pragma omp atomic
            gradients[idx_a + d] += gradient_component;
            #pragma omp atomic
            gradients[idx_n + d] -= gradient_component;
        }
    }
}

void adam_update(std::vector<float>& embedding, const std::vector<float>& gradients,
                 std::vector<float>& m, std::vector<float>& v, int iter, float learning_rate,
                 float beta1, float beta2, float eps) {

    // Bias correction terms (Adam algorithm)
    float beta1_pow = std::pow(beta1, iter + 1);
    float beta2_pow = std::pow(beta2, iter + 1);

    // Parallel Adam update with adaptive learning rates
    #pragma omp parallel for
    for (size_t i = 0; i < embedding.size(); ++i) {
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];

        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * (gradients[i] * gradients[i]);

        // Compute bias-corrected estimates
        float m_hat = m[i] / (1 - beta1_pow);
        float v_hat = v[i] / (1 - beta2_pow);

        // Update parameters with adaptive learning rate
        embedding[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

float compute_pacmap_loss(const std::vector<float>& embedding, const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components) {

    float total_loss = 0.0f;

    for (const auto& triplet : triplets) {
        size_t idx_a = static_cast<size_t>(triplet.anchor) * n_components;
        size_t idx_n = static_cast<size_t>(triplet.neighbor) * n_components;

        // Compute embedding space distance
        float d_ij = compute_distance(embedding.data() + idx_a,
                                     embedding.data() + idx_n,
                                     n_components, PACMAP_METRIC_EUCLIDEAN);

        // Compute loss based on triplet type
        float triplet_loss;
        switch (triplet.type) {
            case NEIGHBOR:
                triplet_loss = w_n * (d_ij / (10.0f + d_ij));
                break;
            case MID_NEAR:
                triplet_loss = w_mn * (d_ij / (10000.0f + d_ij));
                break;
            case FURTHER:
                triplet_loss = w_f * (1.0f / (1.0f + d_ij));
                break;
        }

        total_loss += triplet_loss;
    }

    return total_loss / static_cast<float>(triplets.size());  // Average loss
}

bool check_convergence(const std::vector<float>& loss_history, float threshold, int window) {
    if (loss_history.size() < window) return false;

    // Check if loss has stabilized over the last 'window' iterations
    float recent_avg = 0.0f, older_avg = 0.0f;
    int window_half = window / 2;

    for (int i = loss_history.size() - window; i < loss_history.size(); ++i) {
        if (i < loss_history.size() - window_half) {
            older_avg += loss_history[i];
        } else {
            recent_avg += loss_history[i];
        }
    }

    older_avg /= window_half;
    recent_avg /= window_half;

    return std::abs(recent_avg - older_avg) < threshold;
}
```

#### 2.5 Complete Three-Phase Optimization Loop (Review-Optimized)

```cpp
// File: pacmap_optimization.h (Complete)

#pragma once

#include "pacmap_model.h"
#include <vector>

// Main optimization function with Adam and three-phase weights
void optimize_embedding(PacMapModel* model, float* embedding_out, uwot_progress_callback_v2 callback);

// Initialization utilities
void initialize_random_embedding(std::vector<float>& embedding, int n_samples, int n_components, std::mt19937& rng);
void initialize_adam_state(std::vector<float>& m, std::vector<float>& v, size_t size);

// Safety and monitoring
void compute_safety_stats(PacMapModel* model, const std::vector<float>& embedding);
void monitor_optimization_progress(int iter, int total_iters, float loss,
                                  const std::string& phase, uwot_progress_callback_v2 callback);

// Phase detection utilities
std::string get_current_phase(int iter, int phase1_iters, int phase2_iters);
bool should_terminate_early(const std::vector<float>& loss_history, int max_no_improvement = 100);
```

```cpp
// File: pacmap_optimization.cpp (Complete Implementation)

#include "pacmap_optimization.h"
#include "pacmap_gradient.h"
#include "pacmap_triplet_sampling.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

void optimize_embedding(PacMapModel* model, float* embedding_out, uwot_progress_callback_v2 callback) {
    std::vector<float> embedding(model->n_samples * model->n_components);

    // Initialize with random normal distribution (review specification)
    initialize_random_embedding(embedding, model->n_samples, model->n_components, model->rng);

    int total_iters = model->phase1_iters + model->phase2_iters + model->phase3_iters;

    // Initialize Adam optimizer state
    std::vector<float> gradients(embedding.size());
    std::vector<float> m(embedding.size(), 0.0f);  // First moment
    std::vector<float> v(embedding.size(), 0.0f);  // Second moment

    // Loss history for convergence monitoring
    std::vector<float> loss_history;
    loss_history.reserve(total_iters);

    callback("Starting Optimization", 0, 0.0f, nullptr);

    // Main optimization loop with three phases
    for (int iter = 0; iter < total_iters; ++iter) {
        // Get three-phase weights for current iteration
        auto [w_n, w_mn, w_f] = get_weights(iter, total_iters);

        // Compute gradients for all triplets
        compute_gradients(embedding, model->triplets, gradients,
                         w_n, w_mn, w_f, model->n_components);

        // Update embedding using Adam optimizer
        adam_update(embedding, gradients, m, v, iter, model->learning_rate);

        // Monitor progress and compute loss
        if (iter % 50 == 0 || iter == total_iters - 1) {
            float loss = compute_pacmap_loss(embedding, model->triplets,
                                           w_n, w_mn, w_f, model->n_components);
            loss_history.push_back(loss);

            std::string phase = get_current_phase(iter, model->phase1_iters, model->phase2_iters);
            monitor_optimization_progress(iter, total_iters, loss, phase, callback);

            // Early termination check
            if (iter > 200 && should_terminate_early(loss_history)) {
                callback("Early Termination - Converged", iter,
                        static_cast<float>(iter) / total_iters * 100.0f,
                        "Convergence detected");
                break;
            }
        }
    }

    // Compute final safety statistics
    compute_safety_stats(model, embedding);

    // Save embedding in model for transform operations
    model->embedding = embedding;

    // Copy results to output
    std::memcpy(embedding_out, embedding.data(), embedding.size() * sizeof(float));

    callback("Optimization Complete", total_iters, 100.0f, nullptr);
}

void initialize_random_embedding(std::vector<float>& embedding, int n_samples, int n_components, std::mt19937& rng) {
    std::normal_distribution<float> normal_dist(0.0f, 1e-4f);  // Small random values

    for (auto& val : embedding) {
        val = normal_dist(rng);
    }
}

void initialize_adam_state(std::vector<float>& m, std::vector<float>& v, size_t size) {
    std::fill(m.begin(), m.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
}

void compute_safety_stats(PacMapModel* model, const std::vector<float>& embedding) {
    // Compute pairwise embedding distances for safety analysis
    std::vector<float> embedding_distances;

    // Sample distances for efficiency (similar to UMAP approach)
    int sample_size = std::min(model->n_samples, 1000);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = i + 1; j < sample_size; ++j) {
            float dist = compute_distance(embedding.data() + i * model->n_components,
                                        embedding.data() + j * model->n_components,
                                        model->n_components, PACMAP_METRIC_EUCLIDEAN);
            embedding_distances.push_back(dist);
        }
    }

    // Compute percentiles for outlier detection
    std::sort(embedding_distances.begin(), embedding_distances.end());
    model->min_embedding_dist = embedding_distances[0];
    model->p95_embedding_dist = embedding_distances[embedding_distances.size() * 0.95];
}

void monitor_optimization_progress(int iter, int total_iters, float loss,
                                  const std::string& phase, uwot_progress_callback_v2 callback) {
    float progress = static_cast<float>(iter) / total_iters * 100.0f;
    std::string message = phase + " - Loss: " + std::to_string(loss);
    callback(phase.c_str(), iter, progress, message.c_str());
}

std::string get_current_phase(int iter, int phase1_iters, int phase2_iters) {
    if (iter < phase1_iters) {
        return "Phase 1: Global Structure";
    } else if (iter < phase1_iters + phase2_iters) {
        return "Phase 2: Balance";
    } else {
        return "Phase 3: Local Structure";
    }
}

bool should_terminate_early(const std::vector<float>& loss_history, int max_no_improvement) {
    if (loss_history.size() < max_no_improvement) return false;

    // Check if loss has improved in recent iterations
    float recent_min = *std::min_element(loss_history.end() - max_no_improvement, loss_history.end());
    float historical_min = *std::min_element(loss_history.begin(), loss_history.end());

    // Terminate if no significant improvement
    return (recent_min >= historical_min * 0.999);  // 0.1% tolerance
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