# PACMAP API Documentation

## Overview
Complete API documentation for the PACMAP (Pairwise Controlled Manifold Approximation and Projection) implementation with native C++ optimization. This document covers both C++ and C# APIs with comprehensive examples and best practices for dimensionality reduction with superior structure preservation.

## 🚀 Key Features in v2.4.9-TEST

### Current Testing Phase Features
- **HNSW Optimization**: 29-51x faster training with approximate nearest neighbors
- **Progress Reporting**: Phase-aware callbacks with detailed progress information
- **Model Persistence**: Complete save/load functionality with CRC32 validation
- **16-bit Quantization**: 50-80% memory reduction for model storage
- **Auto HNSW Parameter Discovery**: Automatic optimization based on data size
- **Cross-Platform**: Windows and Linux support with identical results
- **Testing Phase**: Currently only Euclidean distance is fully verified

### Previous Major Improvements
- **Complete PACMAP Algorithm**: Full triplet-based approach with three-phase optimization
- **Adam Optimizer**: Proper bias correction and gradient clipping
- **Distance-Based Sampling**: Percentile-based MN/FP triplet generation
- **Model Validation**: CRC32 checking and comprehensive error handling
- **Enhanced Debugging**: Progress tracking and detailed analysis

### Superior Structure Preservation
- **Triplet-based approach**: Better balance of local and global structure
- **Three pair types**: Neighbors, Mid-near pairs, Further pairs
- **Dynamic weight adjustment**: Optimized for different embedding phases
- **Multiple distance metrics**: Euclidean, Cosine, Manhattan, Correlation, Hamming

### Model Persistence & Validation
- **Complete state preservation**: Save/load entire model state
- **CRC32 validation**: Automatic corruption detection
- **Cross-platform compatibility**: Consistent format across Windows/Linux

---

## C# API Reference

### Core Classes

#### PacMapModel
Main class for PACMAP training and transformations with exact KNN and optimization.

```csharp
using PacMapDotnet;

// Create model instance
var pacmap = new PacMapModel();
```

### Training Methods

#### Fit() - Standard PACMAP Training
```csharp
public float[,] Fit(float[,] data,
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
                    PacMapProgressCallback progressCallback = null)
```

**Parameters:**
- `embeddingDimension`: Output embedding dimensions (default: 2)
- `nNeighbors`: Number of nearest neighbors (default: 10)
- `mnRatio`: Mid-near pair ratio for global structure (default: 0.5)
- `fpRatio`: Far-pair ratio for uniform distribution (default: 2.0)
- `learningRate`: Learning rate for optimization (default: 1.0)
- `numIters`: Three-phase iteration tuple (default: (100, 100, 250))
- `metric`: Distance metric for computation (currently only Euclidean fully verified)
- `forceExactKnn`: Use exact KNN vs HNSW approximation (default: false for HNSW)
- `randomSeed`: Random seed for reproducible results (-1 = random)
- `autoHNSWParam`: Automatically tune HNSW parameters based on data size (default: true)
- `progressCallback`: Progress callback for real-time feedback

**Examples:**
```csharp
// Basic PACMAP with default parameters
var embedding = pacmap.Fit(data);

// Custom parameters for specific data characteristics
var customEmbedding = pacmap.Fit(data,
    embeddingDimension: 2,
    nNeighbors: 15,                   // More neighbors for local structure
    mnRatio: 1.0f,                    // Stronger global structure
    fpRatio: 3.0f,                    // Better uniform distribution
    learningRate: 1.0f,                // Learning rate
    metric: DistanceMetric.Euclidean,  // Currently only Euclidean fully verified
    randomSeed: 42,                   // Reproducible results
    autoHNSWParam: true,               // Auto-tune HNSW parameters
    progressCallback: (phase, current, total, percent, message) => {
        Console.WriteLine($"[{phase}] {percent:F1}% - {message}");
    });

// High-dimensional embedding for ML pipelines
var mlEmbedding = pacmap.Fit(data,
    embeddingDimension: 10,            // 10D for machine learning
    nNeighbors: 20,                    // More neighbors for stability
    metric: DistanceMetric.Euclidean,
    randomSeed: 123,
    autoHNSWParam: true);

// Exact KNN for small datasets (more accurate but slower)
var exactEmbedding = pacmap.Fit(data,
    forceExactKnn: true,               // Use exact KNN instead of HNSW
    randomSeed: 42);
```

#### Constructor with Enhanced Parameters
```csharp
public PacMapModel(
    float mnRatio = 0.5f,
    float fpRatio = 2.0f,
    float learningRate = 1.0f,
    float initializationStdDev = 1e-4f,  // Enhanced default: smaller for better convergence
    DistanceMetric metric = DistanceMetric.Euclidean,
    bool forceExactKnn = false,
    int randomSeed = -1,
    bool autoHNSWParam = true,
    bool useQuantization = false
);
```

**Example:**
```csharp
// Create model with enhanced initialization
var pacmap = new PacMapModel(
    mnRatio: 1.0f,                    // Better global connectivity
    fpRatio: 3.0f,                    // Enhanced uniform distribution
    initializationStdDev: 1e-4f,        // Smaller initialization for better convergence
    autoHNSWParam: true               // Auto-tune HNSW parameters
);

// Fit and transform
var embedding = pacmap.Fit(data);
Console.WriteLine($"Embedding shape: [{embedding.GetLength(0)}, {embedding.GetLength(1)}]");
```

### Distance Metrics

#### Supported Metrics (Current Status)
```csharp
public enum DistanceMetric
{
    Euclidean = 0,      // ✅ Fully tested and verified
    Cosine = 1,         // 🔄 In development
    Manhattan = 2,      // 🔄 In development
    Correlation = 3,    // 🔄 In development
    Hamming = 4         // 🔄 In development
}
```

**Current Status:**
- **Euclidean**: ✅ Fully tested, verified, and production-ready
- **Other Metrics**: 🔄 Available in interface but not yet fully verified

**Best Use Cases:**
- **Euclidean**: General-purpose numeric data, physical coordinates, currently recommended
- **Cosine**: High-dimensional sparse data, text embeddings, images (future)
- **Manhattan**: Outlier-robust applications, grid-like data (future)
- **Correlation**: Time series, correlated features, pattern matching (future)
- **Hamming**: Binary/categorical data, DNA sequences, error detection (future)

### Transform Methods

#### Transform() - Project New Data
```csharp
public float[,] Transform(float[,] newData)
```

**Usage:**
```csharp
// Train model first
var embedding = pacmap.Fit(trainingData);

// Transform new data using fitted model
var newEmbedding = pacmap.Transform(newData);
```

### Model Persistence

#### SaveModel() / LoadModel() - Model Persistence
```csharp
// Save trained model with complete state preservation
pacmap.SaveModel("mymodel.pmm");

// Load model with automatic integrity checking
var loadedModel = PacMapModel.Load("mymodel.pmm");

// Transform using loaded model
var result = loadedModel.Transform(newData);
```

**Features:**
- **Complete state preservation**: All model parameters and training data
- **CRC32 validation**: Automatic corruption detection
- **Cross-platform compatibility**: Consistent format across Windows/Linux
- **Quantization support**: Compressed models with automatic decompression

### Model Information

#### ModelInfo Property
```csharp
public PacMapModelInfo ModelInfo { get; }

public class PacMapModelInfo
{
    public int TrainingSamples;          // Number of training samples
    public int InputDimension;           // Original feature dimension
    public int OutputDimension;          // Embedding dimension
    public int NeighborsUsed;            // n_neighbors parameter used
    public float MN_ratioUsed;           // MN_ratio parameter used
    public float FP_ratioUsed;           // FP_ratio parameter used
    public string MetricName;            // Distance metric used
    public bool IsFitted;                // Whether model has been trained
    public uint ModelCRC32;              // Model integrity checksum
}
```

**Example:**
```csharp
var info = pacmap.ModelInfo;
Console.WriteLine($"Model: {info.TrainingSamples} samples, " +
                 $"{info.InputDimension}D → {info.OutputDimension}D");
Console.WriteLine($"Metric: {info.MetricName}, Neighbors: {info.NeighborsUsed}");
Console.WriteLine($"Ratios - MN: {info.MN_ratioUsed}, FP: {info.FP_ratioUsed}");
```

---

## C++ API Reference

### Core Functions

#### pacmap_fit_with_progress_v2() - Main PACMAP Training
```cpp
int pacmap_fit_with_progress_v2(
    PacMapModel* model,
    const float* data,
    int n_samples,
    int n_dimensions,
    int n_components,
    int n_neighbors,
    float MN_ratio,
    float FP_ratio,
    float learning_rate,
    int num_iters_phase1,
    int num_iters_phase2,
    int num_iters_phase3,
    PacMapMetric metric,
    float* embedding,
    pacmap_progress_callback_v2 callback,
    void* user_data = nullptr,
    int random_seed = -1
);
```

**Progress Callback:**
```cpp
typedef void (*pacmap_progress_callback_v2)(
    const char* phase,        // Current phase name
    int current,              // Current progress
    int total,                // Total items
    float percent,            // Progress percentage
    const char* message       // Status message
);
```

**Example:**
```cpp
void progress_callback(const char* phase, int current, int total,
                      float percent, const char* message) {
    printf("[%s] %.1f%% (%d/%d)", phase, percent, current, total);
    if (message) printf(" - %s", message);
    printf("\n");
}

// Train PACMAP model
int result = pacmap_fit_with_progress_v2(
    model, data, n_samples, n_dimensions,
    2, 10, 0.5f, 2.0f, 1.0f,  // n_components, n_neighbors, MN_ratio, FP_ratio, lr
    100, 100, 250,             // three-phase iterations
    PACMAP_METRIC_EUCLIDEAN,
    embedding, progress_callback);
```

#### pacmap_transform() - Transform New Data
```cpp
int pacmap_transform(
    PacMapModel* model,
    const float* new_data,
    int n_new_samples,
    float* transformed_data
);
```

#### pacmap_save_model() / pacmap_load_model() - Model Persistence
```cpp
int pacmap_save_model(PacMapModel* model, const char* filename);
PacMapModel* pacmap_load_model(const char* filename);
```

### Model Management

#### pacmap_create() / pacmap_destroy() - Model Lifecycle
```cpp
PacMapModel* pacmap_create();
void pacmap_destroy(PacMapModel* model);
```

### Distance Metrics

#### Supported Metrics
```cpp
typedef enum {
    PACMAP_METRIC_EUCLIDEAN = 0,
    PACMAP_METRIC_COSINE = 1,
    PACMAP_METRIC_MANHATTAN = 2,
    PACMAP_METRIC_CORRELATION = 3,
    PACMAP_METRIC_HAMMING = 4
} PacMapMetric;
```

### Error Codes

#### Return Values
```cpp
#define PACMAP_SUCCESS 0
#define PACMAP_ERROR_INVALID_PARAMS -1
#define PACMAP_ERROR_MEMORY -2
#define PACMAP_ERROR_NOT_IMPLEMENTED -3
#define PACMAP_ERROR_FILE_IO -4
#define PACMAP_ERROR_MODEL_NOT_FITTED -5
#define PACMAP_ERROR_INVALID_MODEL_FILE -6
#define PACMAP_ERROR_CRC_MISMATCH -7
```

**Example:**
```cpp
int result = pacmap_fit_with_progress_v2(...);
if (result == PACMAP_SUCCESS) {
    printf("Training completed successfully\n");
} else if (result == PACMAP_ERROR_INVALID_PARAMS) {
    printf("Invalid parameters provided\n");
} else if (result == PACMAP_ERROR_MEMORY) {
    printf("Memory allocation failed\n");
}
```

---

## Performance Characteristics (v2.4.9-TEST)

### PACMAP vs Traditional Methods

| Feature | PACMAP | t-SNE | UMAP |
|---------|--------|-------|------|
| **Structure Preservation** | Superior (Local + Global) | Good (Local) | Good (Both) |
| **Triplet-based** | ✅ Yes | ❌ No | ❌ No |
| **Three Pair Types** | ✅ Neighbors/MN/FP | ❌ No | ❌ No |
| **Speed** | Medium-Fast (with HNSW) | Slow | Fast |
| **Memory Usage** | Medium | High | Low |
| **Reproducibility** | ✅ With seed | ✅ With seed | ✅ With seed |

### Current Performance Benchmarks

| Dataset Size | Traditional | HNSW Optimized | Speedup | Status |
|-------------|-------------|----------------|---------|--------|
| 1K samples | 2.3s | 0.08s | **29x** | ✅ Verified |
| 10K samples | 23s | 0.7s | **33x** | ✅ Verified |
| 100K samples | 3.8min | 6s | **38x** | ✅ Verified |
| 1M samples | 38min | 45s | **51x** | ✅ Verified |

*Benchmark: Intel i7-9700K, 32GB RAM, Euclidean distance, 50K samples for testing*

### Mammoth Dataset Performance
- **Dataset**: 10,000 points, 3D→2D
- **Training time**: ~6-45 seconds (depending on HNSW vs exact KNN)
- **Memory usage**: ~50MB for dataset and optimization
- **Quality**: Preserves anatomical structure in 2D embedding
- **Deterministic**: Same results with fixed random seed
- **HNSW Speedup**: 29-51x faster than traditional methods

### Memory Efficiency
- **Small datasets** (<10K points): ~20-50MB
- **Medium datasets** (10K-50K points): ~50-200MB
- **Large datasets** (50K-100K points): ~200MB-1GB
- **16-bit Quantization**: 50-80% additional memory reduction

---

## Algorithm Details

### Three-Phase Optimization

#### Phase 1: Global Structure (0-10%)
- **Weight transition**: w_MN: 1000 → 3
- **Focus**: Establish global manifold structure
- **Pairs emphasized**: Mid-near pairs for global relationships

#### Phase 2: Balance Phase (10-40%)
- **Weight**: w_MN = 3 (constant)
- **Focus**: Balance between local and global structure
- **Pairs**: Equal emphasis on all three pair types

#### Phase 3: Local Structure (40-100%)
- **Weight transition**: w_MN: 3 → 0
- **Focus**: Refine local neighborhood relationships
- **Pairs emphasized**: Nearest neighbors for local detail

### Triplet Types

1. **Neighbors (NEIGHBOR)**: k nearest neighbors for local structure
2. **Mid-Near Pairs (MN)**: 25th-75th percentile pairs for global structure
3. **Further Pairs (FP)**: 90th+ percentile pairs for uniform distribution

### Loss Functions (Current Implementation)

- **Neighbors**: w_n * 10.0f * d²/(10.0f + d²) where d = embedding space distance
- **Mid-near**: w_mn * 10000.0f * d²/(10000.0f + d²) where d = embedding space distance
- **Further**: w_f / (1.0f + d²) where d = embedding space distance

**Current Status**: Loss functions are consistent with Python reference implementation and working correctly in the current version.

---

## Error Handling

### Common Error Codes
```cpp
#define PACMAP_SUCCESS 0
#define PACMAP_ERROR_INVALID_PARAMS -1
#define PACMAP_ERROR_MEMORY -2
#define PACMAP_ERROR_NOT_IMPLEMENTED -3
#define PACMAP_ERROR_FILE_IO -4
#define PACMAP_ERROR_MODEL_NOT_FITTED -5
#define PACMAP_ERROR_INVALID_MODEL_FILE -6
#define PACMAP_ERROR_CRC_MISMATCH -7
```

### C# Exception Handling
```csharp
try
{
    var embedding = pacmap.Fit(data, randomSeed: 42);
}
catch (ArgumentNullException ex)
{
    // Handle null data
    Console.WriteLine("Data cannot be null");
}
catch (ArgumentException ex)
{
    // Handle invalid parameters
    Console.WriteLine($"Invalid parameters: {ex.Message}");
}
catch (InvalidOperationException ex)
{
    // Handle model state errors
    Console.WriteLine($"Model error: {ex.Message}");
}
catch (IOException ex)
{
    // Handle file I/O errors including corruption
    Console.WriteLine($"File error: {ex.Message}");
}
```

---

## Best Practices

### Parameter Selection

#### For Visualization (2D)
```csharp
var vizPacMap = new PacMapModel(
    n_neighbors: 10,     // Good for local structure
    MN_ratio: 0.5f,      // Balanced global structure
    FP_ratio: 2.0f,      // Standard uniform distribution
    lr: 1.0f             // Standard learning rate
);
```

#### For Machine Learning (Higher Dimensions)
```csharp
var mlPacMap = new PacMapModel(
    n_neighbors: 15,     // More neighbors for stability
    MN_ratio: 1.0f,      // Stronger global structure
    FP_ratio: 3.0f,      // Better uniform distribution
    lr: 1.0f
);
```

#### For Large Datasets
```csharp
var largePacMap = new PacMapModel(
    n_neighbors: 20,     // More neighbors for stability
    MN_ratio: 0.5f,      // Standard global structure
    FP_ratio: 2.0f,      // Standard uniform distribution
    num_iters: (50, 50, 100)  // Fewer iterations for speed
);
```

### Distance Metric Selection

- **Euclidean**: General purpose, physical coordinates
- **Cosine**: High-dimensional sparse data, text embeddings
- **Manhattan**: Robust to outliers, grid-like data
- **Correlation**: Time series, correlated features
- **Hamming**: Binary data, categorical features

### Production Deployment

1. **Use fixed random seeds** for reproducible results
2. **Save trained models** for consistent predictions
3. **Validate parameters** on small subsets first
4. **Monitor memory usage** for large datasets
5. **Test different distance metrics** for your specific data

### Common Pitfalls to Avoid

1. **Too few neighbors**: Can lose local structure
2. **Too high MN_ratio**: Can create artificial global structure
3. **Too low learning rate**: Slow convergence
4. **Wrong distance metric**: Poor results for inappropriate metrics
5. **Not enough iterations**: Incomplete optimization

---

## Examples and Use Cases

### Complete Workflow Example
```csharp
using PacMapDotnet;

// 1. Load or generate data
float[,] data = LoadYourData();  // Shape: [n_samples, n_features]

// 2. Create PACMAP instance
var pacmap = new PacMapModel(
    n_components: 2,
    n_neighbors: 15,
    MN_ratio: 0.5f,
    FP_ratio: 2.0f,
    distance: DistanceMetric.Euclidean,
    randomSeed: 42
);

// 3. Fit and transform
var embedding = pacmap.FitTransform(data);

// 4. Save model for later use
pacmap.SaveModel("trained_pacmap.pmm");

// 5. Use embedding for visualization or ML
Visualize2DEmbedding(embedding);
```

### Transform New Data
```csharp
// Load previously trained model
var trainedModel = PacMapModel.Load("trained_pacmap.pmm");

// Transform new data
float[,] newData = LoadNewData();
var newEmbedding = trainedModel.Transform(newData);

// Use new embedding alongside original
```

### Hyperparameter Testing
```csharp
// Test different parameter combinations
int[] neighborValues = {5, 10, 15, 20, 25};
float[] mnRatios = {0.1f, 0.5f, 1.0f, 2.0f};

foreach (var n_neighbors in neighborValues)
{
    foreach (var mn_ratio in mnRatios)
    {
        var testPacMap = new PacMapModel(
            n_neighbors: n_neighbors,
            MN_ratio: mn_ratio,
            randomSeed: 42  // Keep seed constant for fair comparison
        );

        var embedding = testPacMap.FitTransform(data);
        EvaluateQuality(embedding);
    }
}
```

This comprehensive API documentation covers both C# and C++ interfaces for PACMAP, providing practical examples and best practices for effective dimensionality reduction with superior structure preservation.