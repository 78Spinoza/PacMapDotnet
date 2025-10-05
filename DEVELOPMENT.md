# PACMAPCSharp Development Guide

## Overview

This document provides comprehensive research and analysis for migrating the UMAPCSharp project to implement PACMAP (Pairwise Controlled Manifold Approximation and Projection). The goal is to reuse 95% of the existing infrastructure while implementing PACMAP's unique triplet-based approach.

## Implementation Status Update (2025-10-05)

### Migration Analysis Complete ✅
Comprehensive analysis of the existing PacMapDotnet codebase reveals the current implementation status:

#### File Renaming: COMPLETED ✅
- **35 C++ files** successfully renamed from `uwot_*` to `pacmap_*`
- Directory structure fully prepared for PACMAP implementation
- All files in `C:\PacMapDotnet\src\pacmap_pure_cpp\` show PACMAP naming convention

#### Core Algorithm: NOT IMPLEMENTED ❌
**Critical Finding**: Despite file renaming, both C# and C++ code still contain UMAP implementation
- **C# Code**: Contains UMapModel class with UMAP parameters (minDist, spread)
- **C++ Code**: Contains UMAP fuzzy simplicial set algorithm, not PACMAP triplets
- **Missing Core PACMAP Features**:
  - Triplet sampling (neighbors, mid-near, far pairs)
  - Three-phase dynamic weight optimization
  - PACMAP-specific gradient computation
  - PACMAP parameters (MN_ratio, FP_ratio vs UMAP's minDist, spread)

#### Infrastructure: READY ✅
All supporting infrastructure is in place and ready for PACMAP:
- HNSW optimization for ultra-fast neighbor search
- Complete distance metrics implementation
- Model persistence and quantization systems
- Cross-platform build configuration
- Progress reporting and error handling

#### KNN Performance Issues Identified ⚠️
**Critical Performance Concern**: Analysis of Python reference code reveals major KNN implementation issues:

**Issue 1: Incomplete Python Reference Implementation**
- The `_sample_MN_pair()` and `_sample_FP_pair()` functions are empty stubs (contain only `pass`)
- Only `_sample_neighbors_pair()` is implemented using sklearn's `NearestNeighbors`
- This explains why the user sees inconsistencies - the reference implementation is incomplete

**Issue 2: Direct KNN vs HNSW Optimization Gap**
- Current neighbor sampling uses sklearn's `NearestNeighbors.kneighbors()` (brute force by default)
- No HNSW optimization present in Python reference for triplet sampling
- **Performance Impact**: This would make PACMAP slower than UMAP, violating the requirement

**Issue 3: Triplet Sampling Performance Challenge**
- PACMAP requires 3x more pair sampling than UMAP (neighbors + mid-near + far pairs)
- Without HNSW optimization, PACMAP KNN will be significantly slower than UMAP
- **Critical Requirement**: PACMAP KNN must be faster than UMAP, not slower

**Solution Strategy: HNSW-Optimized Triplet Sampling**
1. Replace sklearn `NearestNeighbors` with HNSW for all pair types
2. Implement distance-based sampling using HNSW distance estimates
3. Optimize triplet generation to exceed UMAP performance

## Algorithm Comparison: UMAP vs PACMAP

### Fundamental Differences

| Aspect | UMAP | PACMAP |
|--------|------|--------|
| **Core Approach** | Fuzzy simplicial sets with 1D topological constraints | Triplet-based structure preservation |
| **Initialization** | Random | Random (no PCA in our implementation) |
| **Pair Types** | Single fuzzy graph | Three distinct pair types: neighbors, mid-near, further |
| **Loss Function** | Cross-entropy on fuzzy sets | Explicit weighted triplet loss |
| **Optimization** | Static parameters | Three-phase dynamic weight adjustment |
| **Local vs Global** | Balanced through min_dist parameter | Explicit control via MN_ratio and FP_ratio |

### Mathematical Formulation

#### UMAP Loss Function
```
L_UMAP = -Σ_ij P_ij log(Q_ij) + Σ_ij (1 - P_ij) log(1 - Q_ij)
where P_ij = fuzzy set membership, Q_ij = low-dimensional similarity
```

#### PACMAP Loss Function
```
L_PACMAP = Σ_neighbors w_n * (d_ij / (10 + d_ij)) +
           Σ_mid-near w_mn * (d_ij / (10000 + d_ij)) +
           Σ_further w_fp * (1 / (1 + d_ij))
where d_ij = embedding space distance
```

### Three-Phase Weight Schedule (PACMAP)

The PACMAP algorithm uses a sophisticated three-phase optimization strategy:

**Phase 1 (0-10% iterations): Global Structure Focus**
- `w_MN` decreases from 1000 to 3
- `w_neighbors = 1.0`, `w_FP = 1.0`
- Emphasis on preserving global manifold structure

**Phase 2 (10-40% iterations): Balance Phase**
- `w_MN = 3.0` (constant)
- `w_neighbors = 1.0`, `w_FP = 1.0`
- Balanced optimization of local and global structure

**Phase 3 (40-100% iterations): Local Structure Focus**
- `w_MN` decreases from 3 to 0
- `w_neighbors = 1.0`, `w_FP = 1.0`
- Final refinement of local neighborhood structure

## Detailed Python Implementation Analysis

### Class Structure

```python
class PaCMAP:
    def __init__(self, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
                 distance="euclidean", lr=1.0, num_iters=(100, 100, 250),
                 verbose=False, apply_pca=True, random_state=None)
```

**Key Parameters:**
- `n_components`: Output embedding dimensions (same as UMAP)
- `n_neighbors`: Number of nearest neighbors (same as UMAP)
- `MN_ratio`: Mid-near pair ratio (PACMAP-specific)
- `FP_ratio`: Far-pair ratio (PACMAP-specific)
- `lr`: Learning rate (PACMAP-specific)
- `num_iters`: Tuple of iterations for three phases (PACMAP-specific)

### Core Algorithm Functions

#### 1. Triplet Sampling (`_sample_triplets`)

```python
def _sample_triplets(self, X):
    # Sample nearest neighbors
    pair_neighbors = self._sample_neighbors_pair(X)

    # Sample mid-near pairs
    n_MN = int(self.n_neighbors * self.MN_ratio)
    pair_MN = self._sample_MN_pair(X, n_MN)

    # Sample further pairs
    n_FP = int(self.n_neighbors * self.FP_ratio)
    pair_FP = self._sample_FP_pair(X, n_FP)

    return pair_neighbors, pair_MN, pair_FP
```

**Analysis:**
- Three separate sampling functions for different pair types
- MN_ratio and FP_ratio control the number of mid-near and far pairs
- Total pairs = n_neighbors + (MN_ratio * n_neighbors) + (FP_ratio * n_neighbors)

#### 2. Neighbor Pair Sampling (`_sample_neighbors_pair`)

```python
def _sample_neighbors_pair(self, X):
    nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric=self.distance).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Remove self (first neighbor)
    pair_neighbors = []
    for i in range(n_samples):
        for j in range(1, self.n_neighbors + 1):
            pair_neighbors.append((i, indices[i, j]))

    return np.array(pair_neighbors)
```

**Implementation Notes for C++:**
- Can reuse HNSW optimization from UMAP for speed
- Need to exclude self-neighbors (index 0 in results)
- Returns pairs as (anchor, neighbor) tuples

#### 3. Weight Schedule (`_find_weight`)

```python
def _find_weight(self, iter, total_iters):
    progress = iter / total_iters

    if progress < 0.1:
        # Phase 1: Global structure (w_MN: 1000 -> 3)
        w_MN = 1000.0 * (1.0 - progress * 10.0) + 3.0 * (progress * 10.0)
    elif progress < 0.4:
        # Phase 2: Balance (w_MN = 3)
        w_MN = 3.0
    else:
        # Phase 3: Local structure (w_MN: 3 -> 0)
        w_MN = 3.0 * (1.0 - (progress - 0.4) / 0.6)

    w_neighbors = 1.0
    w_FP = 1.0
    return w_neighbors, w_MN, w_FP
```

**Key Insights:**
- Linear interpolation within phases
- Smooth transitions between phases
- w_neighbors and w_FP remain constant (only w_MN varies)

#### 4. Gradient Computation (`_pacmap_grad`)

```python
def _pacmap_grad(self, embedding, triplets, w_neighbors, w_MN, w_FP):
    gradients = np.zeros_like(embedding)

    for i, j, triplet_type in triplets:
        diff = embedding[i] - embedding[j]
        d_ij = np.sqrt(np.sum(diff ** 2))

        if triplet_type == 'neighbor':
            # Pull closer: w * 10 / ((10 + d)^2)
            grad_magnitude = w_neighbors * 10.0 / ((10.0 + d_ij) ** 2)
        elif triplet_type == 'mid_near':
            # Moderate pull: w * 10000 / ((10000 + d)^2)
            grad_magnitude = w_MN * 10000.0 / ((10000.0 + d_ij) ** 2)
        else:  # further
            # Push apart: -w / ((1 + d)^2)
            grad_magnitude = -w_FP / ((1.0 + d_ij) ** 2)

        gradient = grad_magnitude * diff / d_ij
        gradients[i] += gradient
        gradients[j] -= gradient

    return gradients
```

**Mathematical Analysis:**
- Different loss functions for each pair type
- Attractive forces for neighbors and mid-near pairs
- Repulsive forces for further pairs
- Gradients are symmetric (equal and opposite for each pair)

## Migration Strategy

### Phase 1: Infrastructure Preparation

1. **Directory Structure** ✅
   - Copy UMAP project to PacMAN
   - Rename main directories
   - Download Python reference files

2. **File Renaming** (In Progress)
   - `uwot_*` → `pacmap_*` (C++ files)
   - `UMap*` → `PacMap*` (C# files)
   - Update namespace references

3. **API Migration**
   - Remove UMAP-specific parameters (`minDist`, `spread`)
   - Add PACMAP-specific parameters (`MN_ratio`, `FP_ratio`, `lr`, `num_iters`)
   - Keep existing HNSW and quantization features

### Phase 2: Core Algorithm Implementation

#### C++ Data Structures

```cpp
// Replace UMAP fuzzy simplicial sets
enum TripletType {
    NEIGHBOR = 0,
    MID_NEAR = 1,
    FURTHER = 2
};

struct Triplet {
    int anchor_idx;
    int neighbor_idx;
    TripletType type;
    float weight;
};

// Replace UwotModel with PacMapModel
struct PacMapModel {
    std::vector<Triplet> neighbor_triplets;
    std::vector<Triplet> mid_near_triplets;
    std::vector<Triplet> further_triplets;

    // Keep HNSW optimization
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> ann_index;

    // PACMAP-specific parameters
    float MN_ratio;
    float FP_ratio;
    float lr;
    std::tuple<int, int, int> num_iters;
};
```

#### Core Functions to Implement

1. **Triplet Sampling Functions**
   ```cpp
   void sample_neighbors_pair(float* data, int n_samples, int n_dims,
                              int n_neighbors, std::vector<Triplet>& triplets);
   void sample_MN_pair(float* data, int n_samples, int n_dims,
                       int n_MN, std::vector<Triplet>& triplets);
   void sample_FP_pair(float* data, int n_samples, int n_dims,
                       int n_FP, std::vector<Triplet>& triplets);
   ```

2. **Weight Schedule Function**
   ```cpp
   std::tuple<float, float, float> find_weight(int iter, int total_iters);
   ```

3. **Gradient Computation**
   ```cpp
   void pacmap_grad(float* embedding, std::vector<Triplet>& triplets,
                    float w_neighbors, float w_MN, float w_FP,
                    float* gradients, int n_samples, int n_dims);
   ```

4. **Optimization Loop**
   ```cpp
   void optimize_embedding(float* embedding, std::vector<Triplet>& triplets,
                           int total_iters, float lr);
   ```

### Phase 3: C# API Migration

#### Updated UMapModel → PacMapModel

```csharp
public class PacMapModel : IDisposable
{
    public float[,] Fit(float[,] data,
        int embeddingDimension = 2,
        int n_neighbors = 10,        // Same as UMAP
        float MN_ratio = 0.5f,       // NEW: Mid-near ratio
        float FP_ratio = 2.0f,       // NEW: Far-pair ratio
        int n_iters = 1000,          // NEW: Total iterations
        DistanceMetric distance = DistanceMetric.Euclidean,
        float lr = 1.0f,             // NEW: Learning rate
        bool forceExactKnn = false,  // Keep HNSW optimization
        bool useQuantization = false, // Keep quantization
        int randomSeed = -1);        // Keep random seed support

    // Keep all existing methods: Transform, SaveModel, etc.
}
```

## Performance Considerations

### Advantages of Using UMAP Infrastructure

1. **HNSW Optimization**: 50-2000x speedup for neighbor finding
2. **Memory Efficiency**: 80-85% reduction vs brute force
3. **Model Persistence**: Save/load with CRC32 validation
4. **Safety Features**: Outlier detection and confidence scoring
5. **Cross-Platform**: Windows/Linux compatibility

### Expected Performance Impact

| Operation | UMAP | PACMAP (Expected) | Impact |
|-----------|------|-------------------|---------|
| **Neighbor Finding** | HNSW optimized | HNSW optimized | No change |
| **Pair Sampling** | Simple graph | Three pair types | 2-3x slower |
| **Gradient Computation** | O(n²) pairs | O(total_triplets) | Similar |
| **Optimization** | Static weights | Dynamic weights | Minimal overhead |
| **Overall** | Baseline | Similar | 5-10% slower max |

## Risk Assessment and Mitigation

### High-Risk Areas

1. **Triplet Sampling Complexity**: PACMAP's three-pair approach is more complex than UMAP's single graph
   - **Mitigation**: Use HNSW for neighbor finding, implement efficient distance-based sampling

2. **Weight Schedule Implementation**: Three-phase optimization requires careful implementation
   - **Mitigation**: Direct translation from Python reference, extensive testing

3. **Memory Usage**: Storing three triplet sets instead of one graph
   - **Mitigation**: Optimize data structures, reuse HNSW for neighbor pairs

### Medium-Risk Areas

1. **Gradient Computation**: Different loss functions require new gradient calculations
   - **Mitigation**: Mathematical validation against Python implementation

2. **Convergence Behavior**: PACMAP may converge differently than UMAP
   - **Mitigation**: Parameter tuning, extensive validation testing

## Validation Strategy

### Unit Testing

1. **Triplet Sampling Tests**
   - Verify neighbor pairs match HNSW results
   - Validate mid-near and far pair distributions
   - Test edge cases (small datasets, high dimensions)

2. **Gradient Computation Tests**
   - Compare gradient values with Python implementation
   - Validate numerical stability
   - Test different triplet types

3. **Weight Schedule Tests**
   - Verify three-phase transitions
   - Test boundary conditions
   - Validate smooth interpolation

### Integration Testing

1. **End-to-End Pipeline Tests**
   - Compare embeddings with Python reference
   - Validate convergence behavior
   - Test different parameter combinations

2. **Performance Tests**
   - Benchmark against UMAP performance
   - Validate HNSW speedup is maintained
   - Test memory usage patterns

3. **Cross-Platform Tests**
   - Windows/Linux compatibility
   - Different dataset sizes
   - Various distance metrics

## KNN Performance Optimization Strategy

### Performance Requirements Analysis

**Critical Success Factor**: PACMAP KNN must be faster than UMAP, not slower, despite requiring 3x more triplet sampling.

### Current Performance Bottlenecks Identified

1. **Triplet Sampling Overhead**: PACMAP requires 3 types of pairs vs UMAP's 1 type
2. **Distance Calculations**: Mid-near and far pairs require distance-based sampling
3. **Lack of Optimization**: Python reference uses brute-force sklearn NearestNeighbors

### HNSW-Optimized Triplet Sampling Strategy

#### 1. Neighbor Pairs (Already Optimized in UMAP)
```cpp
// Use existing HNSW infrastructure from UMAP
// O(log n) per query vs O(n) for brute force
void sample_neighbors_pair(HNSWIndex* index, std::vector<Triplet>& triplets);
```

#### 2. Mid-Near Pairs (HNSW-Optimized)
```cpp
// Strategy 1: Use HNSW distance estimates for distance-based sampling
// Strategy 2: Multi-probe HNSW to sample points at specific distance ranges
void sample_MN_pair_optimized(HNSWIndex* index, float target_distance_range,
                            std::vector<Triplet>& triplets);
```

#### 3. Far Pairs (HNSW-Optimized)
```cpp
// Strategy 1: Sample from HNSW leaf nodes (distant points)
// Strategy 2: Use HNSW to find anti-neighbors (maximally distant)
void sample_FP_pair_optimized(HNSWIndex* index, std::vector<Triplet>& triplets);
```

### Performance Optimization Techniques

#### 1. Distance Estimation Caching
```cpp
// Cache HNSW distance estimates to avoid redundant calculations
struct DistanceCache {
    std::unordered_map<int, float> cached_distances;
    float get_distance(int idx, HNSWIndex* index);
};
```

#### 2. Batch Processing
```cpp
// Process multiple triplet samples in parallel
void batch_sample_triplets(HNSWIndex* index, int batch_size,
                          std::vector<Triplet>& output);
```

#### 3. Adaptive Sampling Strategy
```cpp
// Adjust sampling strategy based on dataset size
enum SamplingStrategy {
    EXACT_SMALL_DATASETS,     // < 10K samples: exact calculations
    HNSW_MEDIUM_DATASETS,     // 10K-100K samples: HNSW optimized
    APPROXIMATE_LARGE_DATASETS // > 100K samples: approximate sampling
};
```

### Expected Performance Gains

| Sampling Method | Brute Force | HNSW Optimized | Speedup |
|----------------|-------------|----------------|---------|
| **Neighbor Pairs** | O(n²) | O(n log n) | 50-200x |
| **Mid-Near Pairs** | O(n²) | O(n log n) | 100-500x |
| **Far Pairs** | O(n²) | O(n log n) | 200-1000x |
| **Overall PACMAP** | O(3n²) | O(3n log n) | **100-300x vs Brute Force** |

### Implementation Plan for KNN Optimization

#### Phase 1: HNSW Integration (Week 2)
- [ ] Implement HNSW-based neighbor pair sampling (reuse from UMAP)
- [ ] Create HNSW-based distance estimation for mid-near pairs
- [ ] Implement HNSW-based far pair sampling using anti-neighbor search

#### Phase 2: Performance Optimization (Week 3)
- [ ] Add distance caching to eliminate redundant calculations
- [ ] Implement batch processing for parallel triplet sampling
- [ ] Create adaptive sampling strategy based on dataset size

#### Phase 3: Validation (Week 4)
- [ ] Benchmark PACMAP vs UMAP KNN performance
- [ ] Validate that PACMAP KNN is faster than UMAP
- [ ] Test accuracy of HNSW-optimized triplet sampling

## Revised Implementation Timeline (Updated 2025-10-05)

### Week 1: Infrastructure and Research ✅
- [x] Copy and rename project structure
- [x] Download Python reference files
- [x] Create comprehensive development documentation
- [x] Analyze Python implementation in detail
- [x] **NEW**: Complete migration status analysis
- [x] **NEW**: Identify critical KNN performance optimization requirements

### Week 2: Core C++ Implementation + KNN Optimization (CURRENT)
- [ ] **CRITICAL**: Replace UMAP algorithm with PACMAP in C++ files
- [ ] **CRITICAL**: Implement HNSW-optimized triplet sampling for all three pair types
- [ ] Implement three-phase weight schedule optimization
- [ ] Implement PACMAP-specific gradient computation
- [ ] **PERFORMANCE FOCUS**: Ensure PACMAP KNN faster than UMAP KNN
- [ ] Create unit tests for all core PACMAP functions

### Week 3: C# API Migration and Integration
- [ ] **HIGH PRIORITY**: Replace UMapModel with PacMapModel in C#
- [ ] Update API parameters (remove minDist/spread, add MN_ratio/FP_ratio/lr)
- [ ] Integrate C++ PACMAP implementation with C# wrapper
- [ ] Implement optimization loop with three-phase weights
- [ ] Test HNSW optimization with PACMAP triplet sampling

### Week 4: Performance Optimization and Validation
- [ ] Benchmark PACMAP vs UMAP performance across datasets
- [ ] Validate embeddings against Python reference implementation
- [ ] Optimize KNN performance to exceed UMAP speed
- [ ] Cross-platform testing and final validation

## Success Criteria

1. **Functional Correctness**: Embeddings match Python reference within acceptable tolerance
2. **Performance**: Maintain 50-2000x HNSW speedup, overall performance within 10% of UMAP
3. **API Compatibility**: Seamless migration from UMAP with minimal code changes
4. **Code Quality**: All tests pass, comprehensive documentation, clean codebase
5. **Cross-Platform**: Works on Windows and Linux with identical results

## References

1. [PaCMAP Python Implementation](https://github.com/YingfanWang/PaCMAP)
2. [PaCMAP Paper](https://arxiv.org/abs/2012.06005)
3. [UMAPCSharp Implementation](https://github.com/78Spinoza/UMAP)
4. [HNSW Library](https://github.com/nmslib/hnswlib)

This development guide provides the foundation for implementing PACMAPCSharp while leveraging the excellent infrastructure built for UMAPCSharp.