PACMAP C++ Native Implementation
================================

Version: 2.0.5-EXACT-KNN-FIX
Author: PacMapDotnet Project
License: MIT

Overview
--------

This is the native C++ implementation of PACMAP (Pairwise Controlled Manifold Approximation and Projection) that provides the core algorithm functionality for the .NET/C# wrapper. The implementation includes complete algorithm fidelity to the Python reference with additional production-ready optimizations.

Key Features
------------

✅ PYTHON-STYLE EXACT KNN: Fixed neighbor sampling to match sklearn behavior exactly
✅ ADAM OPTIMIZER: Full implementation with bias correction and gradient clipping
✅ THREE-PHASE OPTIMIZATION: Correct weight transitions (1000→3→0)
✅ DISTANCE-BASED TRIPLET SAMPLING: Percentile-based MN/FP pair generation
✅ MULTIPLE DISTANCE METRICS: Euclidean, Cosine, Manhattan, Correlation, Hamming
✅ MODEL PERSISTENCE: Complete save/load functionality with CRC32 validation
✅ PARALLEL PROCESSING: OpenMP support for multi-core optimization
✅ CROSS-PLATFORM: Windows and Linux support

File Structure
--------------

Core Implementation Files:
  pacmap_simple_wrapper.h/cpp    - C API interface for C# integration
  pacmap_fit.cpp                 - Core fitting algorithm with triplet sampling
  pacmap_transform.cpp           - New data transformation using fitted models
  pacmap_optimization.cpp        - Three-phase optimization with Adam
  pacmap_gradient.cpp            - Loss function and gradient computation
  pacmap_triplet_sampling.cpp    - Distance-based triplet sampling
  pacmap_distance.h              - Distance metric implementations
  pacmap_utils.h                 - Utility functions and validation

Build System:
  CMakeLists.txt                 - Cross-platform build configuration

Key Implementation Details
--------------------------

1. EXACT KNEIGHBOR SAMPLING (FIXED)
   - Python sklearn-style behavior: k+1 neighbors, skip self
   - Brute-force O(n²) implementation for exact results
   - Parallel processing with OpenMP
   - Identical results to Python reference

2. THREE-PHASE WEIGHT SCHEDULE
   - Phase 1 (0-10%): Global structure (w_mn: 1000→3)
   - Phase 2 (10-40%): Balance phase (w_mn = 3)
   - Phase 3 (40-100%): Local structure (w_mn: 3→0)
   - Fixed weight values: w_n = 1.0f (not 3.0f)

3. ADAM OPTIMIZER
   - Proper bias correction (β₁=0.9, β₂=0.999)
   - Gradient clipping for stability
   - Learning rate adaptation
   - Numerical stability improvements

4. TRIPLET TYPES
   - NEIGHBOR: k nearest neighbors (local structure)
   - MID_NEAR: 25th-75th percentile pairs (global structure)
   - FURTHER: 90th+ percentile pairs (uniform distribution)

5. LOSS FUNCTIONS
   - Consistent with gradient formulas
   - NEW: w_n * 10.0f * d²/(10.0f + d²) for neighbors
   - NEW: w_mn * 10000.0f * d²/(10000.0f + d²) for mid-near
   - Consistent: w_f / (1.0f + d²) for further pairs

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

C API Interface
---------------

Primary Functions:
  pacmap_create()                             - Create new model
  pacmap_destroy(model)                       - Destroy model
  pacmap_fit_with_progress_v2(...)           - Main fitting function
  pacmap_transform(model, new_data, ...)      - Transform new data
  pacmap_save_model(model, filename)          - Save trained model
  pacmap_load_model(filename)                 - Load saved model

Progress Callback:
  typedef void (*pacmap_progress_callback_v2)(
      const char* phase,        // Current phase name
      int current,              // Current progress
      int total,                // Total items
      float percent,            // Progress percentage
      const char* message       // Status message
  );

Distance Metrics:
  PACMAP_METRIC_EUCLIDEAN = 0
  PACMAP_METRIC_COSINE = 1
  PACMAP_METRIC_MANHATTAN = 2
  PACMAP_METRIC_CORRELATION = 3
  PACMAP_METRIC_HAMMING = 4

Error Codes:
  PACMAP_SUCCESS = 0
  PACMAP_ERROR_INVALID_PARAMS = -1
  PACMAP_ERROR_MEMORY = -2
  PACMAP_ERROR_NOT_IMPLEMENTED = -3
  PACMAP_ERROR_FILE_IO = -4
  PACMAP_ERROR_MODEL_NOT_FITTED = -5
  PACMAP_ERROR_INVALID_MODEL_FILE = -6
  PACMAP_ERROR_CRC_MISMATCH = -7

Performance Characteristics
---------------------------

Mammoth Dataset (10,000 points, 3D→2D):
  - Exact KNN: ~2-3 minutes with 450 iterations
  - Memory Usage: ~50MB for dataset and optimization
  - Quality: Preserves anatomical structure in 2D embedding
  - Deterministic: Same results with fixed random seed

Scalability:
  - Optimal: 1K-50K points
  - Maximum tested: 100K points (performance degrades with O(n²) KNN)
  - Memory: O(n²) for neighbor graph during training
  - Parallel: Multi-core support via OpenMP

Version History
---------------

v2.0.5-EXACT-KNN-FIX (Current):
  ✅ FIXED: Exact KNN neighbor sampling to match Python sklearn
  ✅ FIXED: Adam optimizer with proper bias correction
  ✅ FIXED: Loss function gradient consistency
  ✅ FIXED: Three-phase weight transitions
  ✅ FIXED: Distance-based triplet sampling
  ✅ ADDED: CRC32 model validation
  ✅ ADDED: Enhanced progress callbacks
  ✅ ADDED: Comprehensive error handling

Algorithm Validation
--------------------

The implementation has been validated against the official Python PaCMAP reference:

- Neighbor Sampling: Python-style exact KNN with skip-self behavior
- Triplet Types: Proper neighbor/MN/FP triplet classification
- Three-Phase Optimization: Correct weight transitions (1000→3→0)
- Adam Optimization: Proper bias correction and gradient updates
- Loss Functions: Consistent with Python reference implementation
- Stability: Deterministic results with fixed seeds

Integration with C#
-----------------

This C++ library is designed to be called from C# via P/Invoke:

1. C# loads the native library (pacmap.dll on Windows, libpacmap.so on Linux)
2. C# calls pacmap_fit_with_progress_v2() for training
3. C# calls pacmap_transform() for new data projection
4. C# calls pacmap_save_model()/pacmap_load_model() for persistence
5. Progress callbacks provide real-time feedback to C# layer

Memory Management
-----------------

- All memory allocation/deallocation handled internally
- C# only needs to manage float arrays for input/output data
- Automatic cleanup on model destruction
- No memory leaks in current implementation
- Thread-safe for concurrent model usage

Known Limitations
-----------------

- Training uses O(n²) exact KNN (not HNSW-optimized)
- Large datasets (>100K points) may be slow for training
- Transform uses linear search through training data
- No GPU acceleration (CPU only)
- No streaming/incremental learning support

Future Improvements
-------------------

- HNSW optimization for training KNN computation
- GPU acceleration for large-scale datasets
- Streaming/incremental learning capabilities
- Additional distance metrics (Mahalanobis, etc.)
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
- HNSW Library: Yury Malkov & Dmitry Yashunin (used in some components)
- Distance Metrics: Various open-source implementations
- Build System: CMake cross-platform configuration