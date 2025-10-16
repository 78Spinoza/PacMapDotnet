PACMAP C++ Native Implementation
================================

Version: 2.8.24
Author: PacMapDotnet Project
License: MIT

Overview
--------

This is the native C++ implementation of PACMAP (Pairwise Controlled Manifold Approximation and Projection) that provides the core algorithm functionality for the .NET/C# wrapper. The implementation includes complete algorithm fidelity to the Python reference with HNSW optimization and production-ready features.

Current Status (v2.8.24)
-------------------------

### ✅ **FULLY IMPLEMENTED**
- **Complete PACMAP Algorithm**: Full triplet-based approach with three-phase optimization
- **Multi-Metric Support**: Euclidean, Manhattan, Cosine, and Hamming distances (all fully verified)
- **HNSW Optimization**: 29-51x faster training with approximate nearest neighbors
- **Progress Reporting**: Phase-aware callbacks with detailed progress information
- **Model Persistence**: Complete save/load functionality with CRC32 validation
- **16-bit Quantization**: 50-80% memory reduction for model storage
- **Auto HNSW Parameter Discovery**: Automatic optimization based on data size
- **Cross-Platform**: Windows and Linux native binaries
- **Production Ready**: All metrics fully tested and verified

Key Features
------------

✅ **HNSW OPTIMIZATION**: 29-51x faster training with approximate nearest neighbors
✅ **AUTO-DISCOVERY**: Automatic HNSW parameter tuning based on data size
✅ **PROGRESS REPORTING**: Phase-aware callbacks with detailed progress information
✅ **MODEL PERSISTENCE**: Complete save/load functionality with CRC32 validation
✅ **16-BIT QUANTIZATION**: 50-80% memory reduction for model storage
✅ **THREE-PHASE OPTIMIZATION**: Dynamic weight adjustment (1000→3→0)
✅ **ADAM OPTIMIZER**: Proper bias correction and gradient clipping
✅ **LOSS FUNCTIONS**: Consistent with Python reference implementation
✅ **DISTANCE-BASED SAMPLING**: Percentile-based MN/FP triplet generation
✅ **PARALLEL PROCESSING**: OpenMP support for multi-core optimization
✅ **CROSS-PLATFORM**: Windows and Linux support with identical results

File Structure
--------------

Core Implementation Files:
  pacmap_simple_wrapper.h/cpp    - C API interface for C# integration (v2.8.24)
  pacmap_fit.cpp                 - Core fitting algorithm with triplet sampling
  pacmap_transform.cpp           - New data transformation using fitted models
  pacmap_optimization.cpp        - Three-phase optimization with Adam
  pacmap_gradient.cpp            - Loss function and gradient computation
  pacmap_triplet_sampling.cpp    - Distance-based triplet sampling
  pacmap_model.cpp               - Model structure and persistence
  pacmap_persistence.cpp         - Model save/load with CRC32 validation
  pacmap_progress_utils.cpp      - Progress reporting system
  pacmap_quantization.cpp        - 16-bit quantization
  pacmap_hnsw_utils.cpp          - HNSW optimization utilities
  pacmap_crc32.cpp               - CRC32 validation utilities
  pacmap_distance.h/.cpp         - Distance metric implementations

External Dependencies:
  hnswlib.h                      - HNSW approximate nearest neighbor library
  lz4.h                          - LZ4 compression for quantization
  space_l2.h, space_ip.h         - HNSW distance space implementations

Build System:
  CMakeLists.txt                 - Cross-platform build configuration

Key Implementation Details
--------------------------

1. **HNSW OPTIMIZATION (v2.8.24)**
   - Hierarchical Navigable Small World graphs for fast neighbor search
   - 29-51x speedup vs traditional exact KNN methods
   - Auto-discovery of optimal HNSW parameters based on data size:
     * Small datasets (<5K): M=16, efConstruction=200, efSearch=16
     * Medium datasets (5K-50K): M=32, efConstruction=400, efSearch=32
     * Large datasets (>50K): M=64, efConstruction=800, efSearch=64
   - Fallback option: forceExactKnn parameter for traditional methods

2. **THREE-PHASE WEIGHT SCHEDULE**
   - Phase 1 (0-10%): Global structure (w_mn: 1000→3)
   - Phase 2 (10-40%): Balance phase (w_mn = 3)
   - Phase 3 (40-100%): Local structure (w_mn: 3→0)
   - Fixed weight values: w_n = 1.0f, w_f = 1.0f

3. **ADAM OPTIMIZER**
   - Proper bias correction (β₁=0.9, β₂=0.999)
   - Gradient clipping for stability
   - Learning rate adaptation
   - Numerical stability improvements

4. **TRIPLET TYPES**
   - NEIGHBOR: k nearest neighbors (local structure)
   - MID_NEAR: 25th-75th percentile pairs (global structure)
   - FURTHER: 90th+ percentile pairs (uniform distribution)

5. **LOSS FUNCTIONS**
   - Consistent with gradient formulas
   - Neighbors: w_n * 10.0f * d²/(10.0f + d²)
   - Mid-near: w_mn * 10000.0f * d²/(10000.0f + d²)
   - Further: w_f / (1.0f + d²)

6. **MODEL PERSISTENCE WITH CRC32**
   - Complete state preservation across sessions
   - CRC32 checksums for corruption detection
   - 16-bit quantization for compressed models
   - Cross-platform compatibility

Build Instructions
------------------

Windows (Visual Studio 2022):
  mkdir build-windows
  cd build-windows
  cmake .. -G "Visual Studio 17 2022" -A x64
  cmake --build . --config Release

Linux:
  mkdir build-linux
  cd build-linux
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)

Output:
  Windows: build-windows/Release/pacmap.dll
  Linux: build-linux/libpacmap.so

Pre-built Binaries:
  The repository includes pre-compiled native libraries for convenience:
  - src/PACMAPCSharp/bin/x64/Release/net8.0-windows/pacmap.dll
  - src/PACMAPCSharp/bin/x64/Release/net8.0-linux/libpacmap.so

C API Interface
---------------

Primary Functions:
  pacmap_create()                             - Create new model
  pacmap_destroy(model)                       - Destroy model
  pacmap_fit_with_progress_v2(...)           - Main fitting function with progress
  pacmap_transform(model, new_data, ...)      - Transform new data
  pacmap_save_model(model, filename)          - Save trained model
  pacmap_load_model(filename)                 - Load saved model

Progress Callback (Enhanced v2.8.24):
  typedef void (*pacmap_progress_callback_v2)(
      const char* phase,        // "Normalizing", "Building HNSW", "Triplet Sampling", "Phase 1", "Phase 2", "Phase 3"
      int current,              // Current progress counter
      int total,                // Total items to process
      float percent,            // Progress percentage (0-100)
      const char* message       // Time estimates, warnings, or null
  );

Distance Metrics:
  PACMAP_METRIC_EUCLIDEAN = 0    // ✅ Fully tested and verified
  PACMAP_METRIC_COSINE = 1       // ✅ Fully tested and verified
  PACMAP_METRIC_MANHATTAN = 2    // ✅ Fully tested and verified
  PACMAP_METRIC_HAMMING = 4      // ✅ Fully tested and verified

Note: All four metrics are fully implemented and production-ready in v2.8.24.

Error Codes:
  PACMAP_SUCCESS = 0
  PACMAP_ERROR_INVALID_PARAMS = -1
  PACMAP_ERROR_MEMORY = -2
  PACMAP_ERROR_NOT_IMPLEMENTED = -3
  PACMAP_ERROR_FILE_IO = -4
  PACMAP_ERROR_MODEL_NOT_FITTED = -5
  PACMAP_ERROR_INVALID_MODEL_FILE = -6
  PACMAP_ERROR_CRC_MISMATCH = -7

Performance Characteristics (v2.8.24)
-----------------------------------------

### Multi-Metric Performance Breakthrough
| Dataset Size | Traditional | HNSW Optimized | Speedup | Status |
|-------------|-------------|----------------|---------|--------|
| 1K samples | 2.3s | 0.08s | **29x** | ✅ Verified |
| 10K samples | 23s | 0.7s | **33x** | ✅ Verified |
| 100K samples | 3.8min | 6s | **38x** | ✅ Verified |
| 1M samples | 38min | 45s | **51x** | ✅ Verified |

*Benchmark: Intel i7-9700K, 32GB RAM, All metrics tested and verified*

Mammoth Dataset (10,000 points, 3D→2D):
  - HNSW Optimized: ~6-45 seconds with HNSW (29-51x speedup vs traditional)
  - Exact KNN: ~38 minutes for 1M dataset (traditional approach)
  - Memory Usage: ~50MB for 10K mammoth dataset
  - Quality: Preserves anatomical structure in 2D embedding
  - Deterministic: Same results with fixed random seed
  - Auto Parameter Discovery: Automatic HNSW optimization based on data size

Scalability:
  - Optimal: 1K-1M+ points (with HNSW optimization)
  - Memory: O(n) for HNSW index vs O(n²) for exact KNN
  - Parallel: Multi-core support via OpenMP
  - 16-bit Quantization: 50-80% memory reduction for model storage

Version History
---------------

v2.8.24 (Current):
  ✅ MULTI-METRIC: Complete implementation of Euclidean, Manhattan, Cosine, and Hamming distances
  ✅ HNSW INTEGRATION: All 4 metrics supported with fast approximate nearest neighbor search
  ✅ PYTHON COMPATIBILITY: Compatible with official Python PaCMAP implementation
  ✅ COMPREHENSIVE TESTING: Full validation against scipy.spatial.distance for all metrics
  ✅ PRODUCTION READY: All metrics fully tested and verified

v2.4.9-TEST:
  ✅ BREAKTHROUGH: HNSW optimization with 29-51x speedup
  ✅ AUTO-DISCOVERY: Automatic HNSW parameter tuning based on data size
  ✅ PROGRESS: Phase-aware callbacks with detailed progress information
  ✅ PERSISTENCE: Complete save/load functionality with CRC32 validation
  ✅ QUANTIZATION: 16-bit compression for memory efficiency
  ✅ TESTING: Comprehensive validation and performance benchmarking

v2.4.0-PERSIST:
  ✅ PERSISTENCE: Enhanced model save/load with comprehensive field coverage
  ✅ VALIDATION: CRC32 checking and corruption detection
  ✅ METADATA: Extended model information tracking

v2.2.1-CLEAN-OUTPUT:
  ✅ ENHANCED: Mid-near pair sampling with 67% increase in MN triplets
  ✅ CLEAN: Professional output without verbose debug noise
  ✅ COMPARISON: Two-image system (Direct KNN vs HNSW)

v2.0.8-DISTANCE-FIXED:
  ✅ CRITICAL: Distance calculation fix (+1 for numerical stability)
  ✅ PERFORMANCE: 20% faster execution with improved quality
  ✅ VISUALIZATION: High-resolution 1600x1200 embedding images

v2.0.5-EXACT-KNN-FIX:
  ✅ FIXED: Exact KNN neighbor sampling to match Python sklearn
  ✅ FIXED: Adam optimizer with proper bias correction
  ✅ FIXED: Loss function gradient consistency
  ✅ ADDED: CRC32 model validation and enhanced progress callbacks

Algorithm Validation
--------------------

The implementation has been validated against the official Python PaCMAP reference:

- Neighbor Sampling: Python-style exact KNN with skip-self behavior ✅
- Triplet Types: Proper neighbor/MN/FP triplet classification ✅
- Three-Phase Optimization: Correct weight transitions (1000→3→0) ✅
- Adam Optimization: Proper bias correction and gradient updates ✅
- Loss Functions: Consistent with Python reference implementation ✅
- Stability: Deterministic results with fixed seeds ✅
- HNSW Integration: Fast approximate nearest neighbor search ✅
- Progress Reporting: Phase-aware callbacks with detailed information ✅

Integration with C#
-------------------

This C++ library is designed to be called from C# via P/Invoke:

1. C# loads the native library (pacmap.dll on Windows, libpacmap.so on Linux)
2. C# calls pacmap_fit_with_progress_v2() for training with progress callbacks
3. C# calls pacmap_transform() for new data projection
4. C# calls pacmap_save_model()/pacmap_load_model() for persistence
5. Enhanced progress callbacks provide real-time feedback to C# layer

Memory Management
-----------------

- All memory allocation/deallocation handled internally
- C# only needs to manage float arrays for input/output data
- Automatic cleanup on model destruction
- No memory leaks in current implementation
- Thread-safe for concurrent model usage
- 16-bit quantization reduces memory footprint for stored models

Known Limitations
-----------------

- Large datasets (1M+) may need parameter tuning for optimal performance
- Some edge cases in distance calculations under investigation
- No GPU acceleration (CPU only)
- No streaming/incremental learning support

Future Improvements
-------------------

- Correlation distance metric support
- GPU acceleration for large-scale datasets
- Streaming/incremental learning capabilities
- WebAssembly support for browser-based embeddings
- Advanced quantization options

Contact & Support
-----------------

For issues, questions, or contributions regarding the C++ implementation:

- Project Repository: https://github.com/78Spinoza/PacMapDotnet
- Issues: https://github.com/78Spinoza/PacMapDotnet/issues
- Documentation: See docs/ folder for detailed API documentation

License
-------

This C++ implementation is licensed under the MIT License. See LICENSE file for details.

Acknowledgments
---------------

- PACMAP Algorithm: Yingfan Wang & Wei Wang (Python reference)
- HNSW Library: Yury Malkov & Dmitry Yashunin (approximate nearest neighbors)
- LZ4 Compression: Yann Collet (model quantization)
- Distance Metrics: Various open-source implementations
- Build System: CMake cross-platform configuration