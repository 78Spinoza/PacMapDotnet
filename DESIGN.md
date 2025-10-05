# PACMAP Design Document

## Architecture Overview

PACMAP (Pairwise Controlled Manifold Approximation and Projection) is a dimensionality reduction algorithm that preserves both local and global structure through triplet sampling and three-phase optimization.

## Core Design Principles

### 1. Embedding Storage Architecture
```
PacMapModel Structure:
├── Model Parameters (n_samples, n_features, n_components, etc.)
├── Training Data Storage (original training data)
├── Training Embedding Storage (final embedding from fit)
├── Random Seed Storage (for reproducibility)
├── CRC Validation (data integrity)
└── Model State (fitted flag, metadata)
```

### 2. Three-Phase Optimization
- **Phase 1** (33.3%): Neighbor structure optimization
- **Phase 2** (33.3%): Mid-near pair optimization
- **Phase 3** (33.4%): Far pair optimization

### 3. Transform Consistency
Transform operations use stored training data and deterministic seeding to ensure consistent results across sessions.

## Key Components

### PacMapModel Structure
```cpp
struct PacMapModel {
    // Basic parameters
    int n_samples;              // Number of training samples
    int n_features;             // Original feature dimensionality
    int n_components;           // Embedding dimensionality (usually 2)
    int n_neighbors;            // Number of neighbors for triplet sampling
    float MN_ratio;             // Number of mid-near pairs per neighbor
    float FP_ratio;             // Number of further pairs per neighbor
    float learning_rate;        // Learning rate for optimization

    // Data storage for consistency
    std::vector<float> training_data;      // Store original training data
    std::vector<float> training_embedding; // Store final embedding from fit

    // Reproducibility
    int random_seed;            // Random seed used during fitting
    bool has_saved_seed;        // Whether seed was explicitly set

    // Model state
    bool is_fitted;             // Whether model has been fitted

    // Data integrity
    uint32_t model_crc;         // CRC of model parameters
    uint32_t data_crc;          // CRC of training data
    uint32_t embedding_crc;     // CRC of embedding data
};
```

### API Functions

#### Core Functions
- `pacmap_create()` - Create new model instance
- `pacmap_destroy()` - Clean up model resources
- `pacmap_fit_with_progress_v2()` - Fit model with progress callbacks
- `pacmap_transform()` - Transform new data using fitted model

#### Persistence Functions
- `pacmap_save_model()` - Save complete model state to file
- `pacmap_load_model()` - Load model state from file with validation

#### Accessor Functions
- `pacmap_get_n_samples()` - Get number of training samples
- `pacmap_get_n_features()` - Get original feature dimensionality
- `pacmap_get_n_components()` - Get embedding dimensionality
- `pacmap_is_fitted()` - Check if model is fitted

## Data Flow

### Fitting Process
1. **Input Validation** - Check parameters and data integrity
2. **Parameter Storage** - Store all model parameters in model structure
3. **Data Storage** - Store training data for later transform operations
4. **Seed Handling** - Store random seed for reproducibility
5. **Three-Phase Optimization** - Perform PACMAP optimization
6. **Embedding Storage** - Store final embedding in model
7. **CRC Calculation** - Compute CRC values for data integrity

### Transform Process
1. **Model Validation** - Check if model is fitted and data dimensions match
2. **Neighbor Search** - Find nearest neighbors in stored training data
3. **Embedding Generation** - Generate embedding based on nearest neighbors
4. **Deterministic Seeding** - Use stored seed + offset for reproducibility

### Save/Load Process
1. **Parameter Serialization** - Save all model parameters
2. **Data Serialization** - Save training data and embedding
3. **CRC Serialization** - Save CRC values for validation
4. **Load Validation** - Verify CRC values on load
5. **State Restoration** - Restore complete model state

## Implementation Strategy

### 1. Modular Design
- Separate header file with clean API
- Implementation file with all functionality
- Clear separation between interface and implementation

### 2. Error Handling
- Comprehensive error codes for all failure modes
- Detailed error messages for debugging
- Graceful handling of edge cases

### 3. Memory Management
- RAII principles with automatic cleanup
- Vector-based storage for automatic memory management
- Clear ownership semantics

### 4. Performance Considerations
- Efficient data structures for large datasets
- Optimized distance calculations
- Minimal memory overhead

### 5. Cross-Platform Compatibility
- Standard C++17 features
- Platform-specific optimizations guarded by preprocessor
- Consistent behavior across platforms

## Build System

### CMake Configuration
```cmake
cmake_minimum_required(VERSION 3.15)
project(pacmap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(pacmap SHARED
    pacmap_embedding_storage.h
    pacmap_embedding_storage.cpp
)

# Windows-specific settings
if(WIN32)
    target_compile_definitions(pacmap PRIVATE
        _CRT_SECURE_NO_WARNINGS
        PACMAP_EXPORTS
    )
    set_target_properties(pacmap PROPERTIES
        OUTPUT_NAME "pacmap"
        PREFIX ""
        SUFFIX ".dll"
    )
endif()
```

## Testing Strategy

### 1. Unit Tests
- Model creation and destruction
- Parameter validation
- Error handling

### 2. Integration Tests
- Complete fit-transform workflow
- Save/load functionality
- Cross-session consistency

### 3. Quantization Tests
- Embedding storage verification
- Seed persistence validation
- Transform consistency checking

## Usage Example

```cpp
// Create model
PacMapModel* model = pacmap_create();

// Prepare data
int n_samples = 1000;
int n_features = 10;
int embedding_dim = 2;
std::vector<float> data(n_samples * n_features);
std::vector<float> embedding(n_samples * embedding_dim);

// Fit model
int result = pacmap_fit_with_progress_v2(
    model, data.data(), n_samples, n_features, embedding_dim,
    10, 2.0f, 1.0f, 1.0f, 100, 30, 30, 40,
    PACMAP_METRIC_EUCLIDEAN, embedding.data(),
    nullptr, 0, -1, -1, -1, 0, 42, 1
);

// Transform new data
std::vector<float> new_data(50 * n_features);
std::vector<float> new_embedding(50 * embedding_dim);
result = pacmap_transform(model, new_data.data(), 50, n_features, new_embedding.data());

// Save model
pacmap_save_model(model, "mymodel.pacmap");

// Cleanup
pacmap_destroy(model);
```

## Future Enhancements

### 1. Performance Optimizations
- Parallel processing for large datasets
- GPU acceleration support
- Memory-mapped file I/O for large models

### 2. Advanced Features
- Custom distance metrics
- Incremental learning support
- Multi-scale embedding

### 3. Integration Features
- C#/.NET wrapper improvements
- Python bindings
- REST API service

## Conclusion

This design provides a robust, efficient, and maintainable PACMAP implementation that addresses the core requirements of embedding storage, seed persistence, and transform consistency while maintaining cross-platform compatibility and performance.