# PacMapDotnet Development Guide

## Overview

This document provides comprehensive technical information about the current PacMapDotnet implementation. The project has successfully migrated from UMAPCSharp to a complete PACMAP (Pairwise Controlled Manifold Approximation and Projection) implementation with production-ready features and optimizations.

## Current Implementation Status (v2.4.9-TEST)

### ✅ **COMPLETED IMPLEMENTATION**

The PACMAP implementation is **fully functional** with the following completed components:

#### **Core Algorithm Implementation**
- ✅ **PACMAP Triplet-based Approach**: Three pair types for superior structure preservation
- ✅ **Three-Phase Optimization**: Dynamic weight adjustment (1000→3→0)
- ✅ **Adam Optimizer**: Proper gradient descent with bias correction and gradient clipping
- ✅ **Loss Functions**: Consistent with Python reference implementation
- ✅ **Distance-Based Sampling**: Percentile-based MN/FP triplet generation
- ✅ **Model Persistence**: Complete save/load functionality with CRC32 validation

#### **Production Features**
- ✅ **C# API**: Comprehensive wrapper with progress callbacks and error handling
- ✅ **Distance Metrics**: Euclidean (fully verified), others in testing
- ✅ **Model Validation**: CRC32 checking and comprehensive error handling
- ✅ **Cross-Platform**: Windows and Linux native binaries
- ✅ **Demo Application**: Complete mammoth dataset with anatomical visualization
- ✅ **HNSW Optimization**: Fast approximate nearest neighbor search
- ✅ **Progress Reporting**: Phase-aware callbacks with detailed progress
- ✅ **16-bit Quantization**: Memory-efficient model storage

#### **Visualization & Testing**
- ✅ **OxyPlot Integration**: 2D embedding visualization with anatomical part coloring
- ✅ **Hyperparameter Testing**: Comprehensive parameter exploration utilities
- ✅ **Anatomical Classification**: Automatic part detection (feet, legs, body, head, trunk, tusks)
- ✅ **3D Visualization**: Multiple views (XY, XZ, YZ) for reference datasets

## Current Architecture

### Core C++ Implementation Files

```
src/pacmap_pure_cpp/
├── pacmap_simple_wrapper.h/cpp      # C API interface
├── pacmap_fit.cpp                   # Core fitting algorithm with triplet sampling
├── pacmap_transform.cpp             # New data transformation using fitted models
├── pacmap_optimization.cpp          # Three-phase optimization with Adam
├── pacmap_gradient.cpp              # Loss function and gradient computation
├── pacmap_triplet_sampling.cpp      # Distance-based triplet sampling
├── pacmap_model.cpp                 # Model structure and persistence
├── pacmap_distance.h                # Distance metric implementations
├── pacmap_utils.h                   # Utility functions and validation
├── pacmap_persistence.cpp           # Model save/load with CRC32 validation
├── pacmap_progress_utils.cpp        # Progress reporting system
├── pacmap_quantization.cpp          # 16-bit quantization
├── pacmap_hnsw_utils.cpp            # HNSW optimization utilities
└── CMakeLists.txt                   # Build configuration
```

### C# Wrapper Implementation

```
src/PACMAPCSharp/
├── PacMapModel.cs                   # Main API class with comprehensive functionality
├── pacmap.dll                       # Native binary (v2.4.9-TEST)
└── PACMAPCSharp.csproj             # Project configuration
```

### Demo and Visualization

```
src/PacMapDemo/
├── Program.cs                       # Main demo with mammoth dataset
├── Program_Complex.cs               # Hyperparameter testing utilities
├── Visualizer.cs                    # OxyPlot-based visualization
├── Data/                            # Dataset directory
└── Gif/                             # Generated visualizations
```

## Algorithm Implementation Details

### PACMAP Loss Function (Current Implementation)

```cpp
// Current loss function implementation
float compute_pacmap_loss(const std::vector<float>& embedding,
                         const std::vector<Triplet>& triplets,
                         float w_n, float w_mn, float w_f, int n_components) {
    float total_loss = 0.0f;

    for (const auto& triplet : triplets) {
        // Compute distance in embedding space
        float dist_squared = 0.0f;
        for (int d = 0; d < n_components; ++d) {
            float diff = embedding[idx_a + d] - embedding[idx_n + d];
            dist_squared += diff * diff;
        }

        // Loss function based on triplet type
        switch (triplet.type) {
            case NEIGHBOR:
                loss_term = w_n * 10.0f * dist_squared / (10.0f + dist_squared);
                break;
            case MID_NEAR:
                loss_term = w_mn * 10000.0f * dist_squared / (10000.0f + dist_squared);
                break;
            case FURTHER:
                loss_term = w_f / (1.0f + dist_squared);
                break;
        }
        total_loss += loss_term;
    }

    return total_loss;
}
```

### Three-Phase Weight Schedule (Current)

```cpp
std::tuple<float, float, float> get_weights(int current_iter, int phase1_end, int phase2_end) {
    float w_n, w_mn, w_f = 1.0f;

    if (current_iter < phase1_end) {
        // Phase 1: Global structure (0-10%): w_mn: 1000→3 transition
        float progress = (float)current_iter / phase1_end;
        w_n = 1.0f;
        w_mn = 1000.0f * (1.0f - progress) + 3.0f * progress;
    } else if (current_iter < phase2_end) {
        // Phase 2: Balance phase (10-40%): stable weights
        w_n = 1.0f;
        w_mn = 3.0f;
    } else {
        // Phase 3: Local structure (40-100%): w_mn: 3→0 transition
        float progress_in_phase3 = (float)(current_iter - phase2_end) / (total_iters - phase2_end);
        w_n = 1.0f;
        w_mn = 3.0f * (1.0f - progress_in_phase3);
    }

    return {w_n, w_mn, w_f};
}
```

## Current C# API

### PacMapModel Class (Current Implementation)

```csharp
public class PacMapModel : IDisposable
{
    // Constructor with enhanced parameters
    public PacMapModel(
        float mnRatio = 0.5f,
        float fpRatio = 2.0f,
        float learningRate = 1.0f,
        float initializationStdDev = 1e-4f,
        DistanceMetric metric = DistanceMetric.Euclidean,
        bool forceExactKnn = false,
        int randomSeed = -1,
        bool autoHNSWParam = true,
        bool useQuantization = false
    );

    // Main fitting method
    public float[,] Fit(
        float[,] data,
        int embeddingDimension = 2,
        int nNeighbors = 10,
        float mnRatio = 0.5f,
        float fpRatio = 2.0f,
        float learningRate = 1.0f,
        (int, int, int) numIters = (100, 100, 250),
        DistanceMetric metric = DistanceMetric.Euclidean,
        bool forceExactKnn = false,
        int randomSeed = -1,
        bool autoHNSWParam = true,
        PacMapProgressCallback progressCallback = null
    );

    // Transform new data
    public float[,] Transform(float[,] newData);

    // Model persistence
    public void SaveModel(string filename);
    public static PacMapModel Load(string filename);

    // Model information
    public PacMapModelInfo ModelInfo { get; }
}
```

### Progress Callback (Current)

```csharp
public delegate void PacMapProgressCallback(
    string phase,        // "Normalizing", "Building HNSW", "Triplet Sampling",
                        // "Phase 1: Global", "Phase 2: Balanced", "Phase 3: Local"
    int current,         // Current progress counter
    int total,           // Total items to process
    float percent,       // Progress percentage (0-100)
    string message       // Time estimates, warnings, or null
);
```

## Performance Characteristics (Current)

### Mammoth Dataset Performance
- **Dataset**: 10,000 points, 3D→2D
- **Training time**: ~6-45 seconds (depending on HNSW vs exact KNN)
- **Memory usage**: ~50MB for dataset and optimization
- **Quality**: Preserves anatomical structure in 2D embedding
- **Deterministic**: Same results with fixed random seed
- **HNSW Speedup**: 29-51x faster than traditional methods

### Recent Performance Improvements
- ✅ **HNSW Optimization**: 29-51x speedup vs traditional methods
- ✅ **Progress Reporting**: Phase-aware callbacks with detailed progress
- ✅ **Model Persistence**: Complete save/load with CRC32 validation
- ✅ **16-bit Quantization**: 50-80% memory reduction
- ✅ **Auto HNSW Parameter Discovery**: Automatic optimization based on data size

## Current Testing Status

### ✅ **WORKING FEATURES**
- **Euclidean Distance**: Fully tested and verified
- **HNSW Optimization**: Fast approximate nearest neighbors
- **Model Persistence**: Save/load with CRC32 validation
- **Progress Reporting**: Phase-aware callbacks with detailed progress
- **16-bit Quantization**: Memory-efficient model storage
- **Cross-Platform**: Windows and Linux support
- **Multiple Dimensions**: 1D to 50D embeddings
- **Transform Capability**: Project new data using fitted models
- **Outlier Detection**: 5-level safety analysis

### 🔄 **IN DEVELOPMENT**
- **Additional Distance Metrics**: Cosine, Manhattan, Correlation, Hamming
- **GPU Acceleration**: CUDA support for large datasets
- **WebAssembly Support**: Browser-based embeddings
- **Streaming Processing**: Large dataset handling

### ⚠️ **KNOWN LIMITATIONS**
- Only Euclidean distance is fully verified
- Large datasets (1M+) may need parameter tuning
- Some edge cases in distance calculations under investigation

## Build Instructions (Current)

### Prerequisites
- **.NET 8.0+**: For C# wrapper compilation
- **Visual Studio Build Tools** (Windows) or **GCC** (Linux)

### Quick Build
```bash
# Clone repository
git clone https://github.com/78Spinoza/PacMapDotnet.git
cd PacMapDotnet

# Build solution
dotnet build src/PACMAPCSharp.sln --configuration Release

# Run demo
cd src/PacMapDemo
dotnet run
```

### Pre-built Binaries
The repository includes pre-compiled native libraries:
- `src/PACMAPCSharp/bin/x64/Release/net8.0-windows/pacmap.dll` (Windows x64)
- `src/PACMAPCSharp/bin/x64/Release/net8.0-linux/libpacmap.so` (Linux x64)

No C++ compilation required for basic usage!

## Demo Application Features

### Complete Demo Application
- ✅ **Mammoth Dataset**: 10,000 point 3D mammoth anatomical dataset
- ✅ **1M Hairy Mammoth**: Large-scale dataset testing capabilities
- ✅ **Anatomical Classification**: Automatic part detection (feet, legs, body, head, trunk, tusks)
- ✅ **3D Visualization**: Multiple views (XY, XZ, YZ) with high-resolution output
- ✅ **PACMAP Embedding**: 2D embedding with anatomical coloring
- ✅ **Hyperparameter Testing**: Comprehensive parameter exploration with GIF generation
- ✅ **Model Persistence**: Save/load functionality testing
- ✅ **Progress Reporting**: Real-time progress tracking with phase-aware callbacks

## Current Architecture Summary

The PacMapDotnet implementation successfully provides:
1. **Complete PACMAP Algorithm**: Full triplet-based approach with three-phase optimization
2. **Production-Ready Features**: Model persistence, HNSW optimization, progress reporting
3. **High Performance**: 29-51x speedup over traditional methods
4. **Cross-Platform Support**: Windows and Linux with identical results
5. **Comprehensive API**: Easy-to-use C# wrapper with full functionality
6. **Extensive Testing**: Validation against Python reference implementation

The implementation is currently in **testing phase (v2.4.9-TEST)** with Euclidean distance fully verified and additional features under active development.

*This development guide reflects the current state of the PacMapDotnet implementation as of version 2.4.9-TEST.*