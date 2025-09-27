# PacMAP Enhanced C# Library - Architecture Design

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    C# Application Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  PacMAPSharp.dll (Type-Safe C# Wrapper)                       │
│  ├─ PacMAPModel.cs           # Main API class                  │
│  ├─ Enums & Types            # DistanceMetric, OutlierLevel   │
│  ├─ TransformResult.cs       # Enhanced result structures      │
│  └─ Cross-Platform P/Invoke  # Windows/Linux/macOS detection  │
├─────────────────────────────────────────────────────────────────┤
│                    C FFI Interface                             │
│  ├─ pacmap_create()          # Model lifecycle management      │
│  ├─ pacmap_fit*()            # Training functions              │
│  ├─ pacmap_transform*()      # Inference functions             │
│  └─ pacmap_save/load()       # Persistence operations          │
├─────────────────────────────────────────────────────────────────┤
│                 Rust Core Engine                               │
│  ├─ lib.rs                   # Main API & FFI exports          │
│  ├─ stats.rs                 # Normalization & statistics      │
│  ├─ pairs.rs                 # HNSW neighbor search            │
│  ├─ quantize.rs              # 16-bit compression              │
│  └─ serialization.rs         # Model persistence               │
├─────────────────────────────────────────────────────────────────┤
│                  External Dependencies                         │
│  ├─ pacmap crate             # Core PacMAP algorithm           │
│  ├─ usearch crate            # HNSW implementation             │
│  ├─ ndarray crate            # Multidimensional arrays         │
│  └─ serde + zstd             # Serialization & compression     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **Performance First**: Native Rust core for computational efficiency
2. **Type Safety**: Strong typing throughout the C# API
3. **Cross-Platform**: Unified interface across Windows/Linux/macOS
4. **Production Ready**: Comprehensive error handling and validation
5. **Memory Efficient**: Optimized data structures and HNSW acceleration
6. **Modular Design**: Clear separation of concerns for maintainability

## 2. Core Components

### 2.1 Rust Engine Architecture

#### 2.1.1 Module Structure
```rust
// lib.rs - Main API and FFI exports
pub mod stats;           // Normalization and statistical analysis
pub mod pairs;           // HNSW-enhanced neighbor search
pub mod quantize;        // 16-bit quantization system
pub mod serialization;   // Model persistence with compression

// Core data structures
pub struct PacMAPModel {
    embedding: Array2<f64>,        // High-precision embedding
    config: Configuration,          // Algorithm parameters
    stats: DistanceStats,          // Training data statistics
    normalization: NormParams,     // Feature scaling parameters
    hnsw_index: Option<HnswIndex>, // HNSW search structure
}
```

#### 2.1.2 HNSW Integration (from UMAP lessons)
```rust
// Inspired by UMAP's HNSW parameter management
pub struct HnswConfig {
    pub m: usize,              // Graph degree (16-64, auto-scaled)
    pub ef_construction: usize, // Build quality (64-512)
    pub ef_search: usize,      // Query speed (32-256)
    pub distance_metric: DistanceMetric,
}

impl HnswConfig {
    // Auto-scaling logic based on dataset size (from UMAP analysis)
    pub fn auto_tune(n_samples: usize, metric: DistanceMetric) -> Self {
        let (m, ef_c) = match n_samples {
            n if n < 1000  => (16, 200),   // Small datasets
            n if n < 10000 => (32, 128),   // Medium datasets
            _ => (64, 256),                // Large datasets
        };

        Self {
            m,
            ef_construction: ef_c,
            ef_search: ef_c / 2,
            distance_metric: metric,
        }
    }
}
```

#### 2.1.3 Normalization System (Inspired by UMAP)
```rust
// Following UMAP's approach to normalization consistency
#[derive(Serialize, Deserialize, Clone)]
pub struct NormalizationParams {
    pub means: Vec<f64>,           // Feature means
    pub stds: Vec<f64>,            // Feature standard deviations
    pub mode: NormalizationMode,   // Z-score, MinMax, Robust
    pub feature_ranges: Vec<(f64, f64)>, // Min/max per feature
}

pub fn normalize_consistent(
    data: &mut Array2<f64>,
    params: &mut NormalizationParams,
    is_training: bool
) -> Result<(), Box<dyn Error>> {
    if is_training {
        // Compute and save normalization parameters
        compute_normalization_params(data, params)?;
    }

    // Apply normalization using saved parameters
    apply_normalization(data, params)?;
    Ok(())
}
```

### 2.2 C# Wrapper Architecture

#### 2.2.1 Platform Detection (from UMAP)
```csharp
public class PacMAPModel : IDisposable
{
    private static readonly bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
    private static readonly bool IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
    private static readonly bool IsMacOS = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);

    // Platform-specific P/Invoke declarations
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static IntPtr CallCreate()
    {
        return IsWindows ? WindowsCreate() :
               IsLinux ? LinuxCreate() :
               MacOSCreate();
    }
}
```

#### 2.2.2 Enhanced Error Handling
```csharp
private static void ThrowIfError(int errorCode)
{
    if (errorCode == PACMAP_SUCCESS) return;

    var message = CallGetErrorMessage(errorCode);

    throw errorCode switch
    {
        PACMAP_ERROR_INVALID_PARAMS => new ArgumentException(message),
        PACMAP_ERROR_MEMORY => new OutOfMemoryException(message),
        PACMAP_ERROR_NOT_IMPLEMENTED => new NotImplementedException(message),
        PACMAP_ERROR_FILE_IO => new IOException(message),
        PACMAP_ERROR_MODEL_NOT_FITTED => new InvalidOperationException(message),
        PACMAP_ERROR_INVALID_MODEL_FILE => new InvalidDataException(message),
        _ => new Exception($"PacMAP Error ({errorCode}): {message}")
    };
}
```

## 3. Data Flow Architecture

### 3.1 Training Pipeline

```
Input Data (C#)
       ↓
1. Validation & Type Conversion
   - Dimension checking
   - NaN/Infinity detection
   - Array flattening
       ↓
2. Rust Core Processing
   - Feature normalization (saved params)
   - HNSW neighbor search
   - PacMAP optimization
   - Distance statistics computation
       ↓
3. Result Processing (C#)
   - 2D array reconstruction
   - Model info extraction
   - Memory cleanup
       ↓
Embedding + Model (C#)
```

### 3.2 Transform Pipeline

```
New Data (C#)
       ↓
1. Validation Against Training
   - Feature dimension matching
   - Model fitted verification
       ↓
2. Rust Transform Processing
   - Apply saved normalization
   - HNSW nearest neighbor search
   - PacMAP projection
   - Distance analysis (optional)
       ↓
3. Enhanced Result Assembly (C#)
   - Embedding coordinates
   - Distance statistics
   - Outlier classification
   - Confidence scoring
       ↓
TransformResult[] (C#)
```

### 3.3 Persistence Pipeline

```
Fitted Model (C#)
       ↓
1. Serialization Preparation (Rust)
   - Model parameter extraction
   - Optional quantization
   - HNSW index serialization
       ↓
2. Compression & Storage
   - Bincode serialization
   - ZSTD compression (3-5x reduction)
   - Endian-safe binary format
       ↓
3. File Management (C#)
   - Directory creation
   - Atomic write operations
   - Error handling
       ↓
Compressed Model File
```

## 4. Memory Management

### 4.1 Rust Memory Model

```rust
// RAII-based resource management
pub struct PacMAPModel {
    // Owned data structures
    embedding: Array2<f64>,
    hnsw_index: Option<Box<dyn HnswIndex>>,

    // Reference-counted shared data
    normalization: Arc<NormalizationParams>,
}

impl Drop for PacMAPModel {
    fn drop(&mut self) {
        // Automatic cleanup of native resources
        // HNSW index destructor called automatically
    }
}
```

### 4.2 C# Memory Model

```csharp
public class PacMAPModel : IDisposable
{
    private IntPtr _nativeModel;
    private bool _disposed = false;

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && _nativeModel != IntPtr.Zero)
        {
            // Call native destructor
            CallDestroy(_nativeModel);
            _nativeModel = IntPtr.Zero;
            _disposed = true;
        }
    }

    ~PacMAPModel() => Dispose(false);
}
```

### 4.3 Memory Optimization Strategies

1. **HNSW Index Sharing**: Reuse indexes for multiple transforms
2. **Lazy Quantization**: Only compress when saving models
3. **Streaming Processing**: Handle large datasets in chunks
4. **Memory Pools**: Reuse temporary buffers for batch operations

## 5. Concurrency Model

### 5.1 Thread Safety

```rust
// Thread-safe components using Arc<Mutex<T>>
pub struct ThreadSafePacMAPModel {
    inner: Arc<Mutex<PacMAPModel>>,
    hnsw_index: Arc<RwLock<HnswIndex>>, // Multiple readers, single writer
}

// Read-only operations can use RwLock for better performance
impl ThreadSafePacMAPModel {
    pub fn transform_batch(&self, data: &Array2<f64>) -> Result<Array2<f64>, Error> {
        let index = self.hnsw_index.read().unwrap();
        // Parallel transform using rayon
        Ok(transform_parallel(data, &*index)?)
    }
}
```

### 5.2 C# Concurrency

```csharp
// Thread-safe wrapper for concurrent access
public class ThreadSafePacMAPModel
{
    private readonly object _lock = new object();
    private readonly PacMAPModel _model;

    public float[,] Transform(float[,] data)
    {
        // Only lock for coordination, not computation
        lock (_lock)
        {
            return _model.Transform(data);
        }
    }
}
```

## 6. Performance Optimizations

### 6.1 HNSW Optimization (from UMAP analysis)

```rust
// Dynamic parameter tuning based on data characteristics
impl HnswOptimizer {
    pub fn optimize_params(
        n_samples: usize,
        n_features: usize,
        target_accuracy: f32
    ) -> HnswConfig {
        // Memory estimation (from UMAP formula)
        let estimated_memory_mb = (n_samples * 32 * 4 * 2) / (1024 * 1024);

        // Auto-scale based on memory constraints
        let m = if estimated_memory_mb > 1000 { 16 } else { 32 };
        let ef_construction = std::cmp::min(200 + n_samples / 100, 512);

        HnswConfig {
            m,
            ef_construction,
            ef_search: ef_construction / 2,
            distance_metric: auto_select_metric(n_features),
        }
    }
}
```

### 6.2 Quantization Strategy

```rust
// Intelligent quantization with quality control
pub struct QuantizationConfig {
    pub enable: bool,
    pub target_accuracy: f32,   // e.g., 0.99 for 99% accuracy retention
    pub compression_level: u8,  // 1-9 for ZSTD
}

impl QuantizationProcessor {
    pub fn quantize_with_validation(
        embedding: &Array2<f64>,
        config: &QuantizationConfig
    ) -> Result<Array2<f16>, QuantizationError> {
        let quantized = quantize_f64_to_f16(embedding);

        // Validate accuracy loss
        let accuracy = compute_accuracy_retention(&embedding, &quantized);
        if accuracy < config.target_accuracy {
            return Err(QuantizationError::AccuracyLoss(accuracy));
        }

        Ok(quantized)
    }
}
```

## 7. Error Handling Strategy

### 7.1 Hierarchical Error Model

```rust
// Comprehensive error hierarchy
#[derive(Debug, Error)]
pub enum PacMAPError {
    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },

    #[error("Memory allocation failed: {size} bytes")]
    MemoryError { size: usize },

    #[error("HNSW index error: {source}")]
    HnswError { source: usearch::Error },

    #[error("Serialization error: {source}")]
    SerializationError { source: serde::Error },

    #[error("IO error: {source}")]
    IoError { source: std::io::Error },
}

// Error code mapping for C FFI
impl PacMAPError {
    pub fn to_error_code(&self) -> i32 {
        match self {
            PacMAPError::InvalidParameters { .. } => -1,
            PacMAPError::MemoryError { .. } => -2,
            PacMAPError::HnswError { .. } => -3,
            PacMAPError::SerializationError { .. } => -4,
            PacMAPError::IoError { .. } => -5,
        }
    }
}
```

### 7.2 C# Exception Mapping

```csharp
// Rich exception hierarchy for C# users
public class PacMAPException : Exception
{
    public int ErrorCode { get; }

    public PacMAPException(int errorCode, string message)
        : base(message)
    {
        ErrorCode = errorCode;
    }
}

public class PacMAPMemoryException : PacMAPException
{
    public long RequestedSize { get; }

    public PacMAPMemoryException(long size, string message)
        : base(-2, message)
    {
        RequestedSize = size;
    }
}
```

## 8. Build and Deployment Architecture

### 8.1 Cross-Platform Build System

```yaml
# GitHub Actions workflow for multi-platform builds
strategy:
  matrix:
    include:
      - os: windows-latest
        target: x86_64-pc-windows-msvc
        artifact: pacmap_enhanced.dll
      - os: ubuntu-latest
        target: x86_64-unknown-linux-gnu
        artifact: libpacmap_enhanced.so
      - os: macos-latest
        target: x86_64-apple-darwin
        artifact: libpacmap_enhanced.dylib

steps:
  - name: Build Rust Library
    run: |
      cargo build --release --target ${{ matrix.target }}

  - name: Build C# Wrapper
    run: |
      dotnet pack PacMAPSharp/PacMAPSharp.csproj

  - name: Test Integration
    run: |
      dotnet test PacMAPSharp.Tests/
```

### 8.2 NuGet Package Structure

```
PacMAPSharp.1.0.0.nupkg
├── lib/net8.0/
│   ├── PacMAPSharp.dll
│   ├── PacMAPSharp.xml      # API documentation
│   └── PacMAPSharp.pdb      # Debug symbols
├── runtimes/
│   ├── win-x64/native/
│   │   └── pacmap_enhanced.dll
│   ├── linux-x64/native/
│   │   └── libpacmap_enhanced.so
│   └── osx-x64/native/
│       └── libpacmap_enhanced.dylib
├── README.md
└── LICENSE
```

## 9. Monitoring and Diagnostics

### 9.1 Performance Metrics

```csharp
public class PacMAPMetrics
{
    public TimeSpan TrainingTime { get; set; }
    public TimeSpan TransformTime { get; set; }
    public long PeakMemoryUsage { get; set; }
    public int HnswIndexSize { get; set; }
    public float AccuracyScore { get; set; }
    public int OutlierCount { get; set; }
}

// Built-in performance monitoring
public static class PacMAPDiagnostics
{
    public static void LogPerformanceMetrics(PacMAPMetrics metrics)
    {
        Console.WriteLine($"Training: {metrics.TrainingTime.TotalMilliseconds}ms");
        Console.WriteLine($"Memory: {metrics.PeakMemoryUsage / 1024 / 1024}MB");
        Console.WriteLine($"Accuracy: {metrics.AccuracyScore:P2}");
    }
}
```

### 9.2 Debug and Validation Tools

```rust
#[cfg(debug_assertions)]
pub fn validate_embedding_quality(
    original: &Array2<f64>,
    embedding: &Array2<f64>
) -> ValidationReport {
    ValidationReport {
        neighbor_preservation: compute_neighbor_preservation(original, embedding),
        global_structure_score: compute_global_structure_score(original, embedding),
        local_structure_score: compute_local_structure_score(original, embedding),
        stress_measure: compute_stress(original, embedding),
    }
}
```

## 10. Future Extensibility

### 10.1 Plugin Architecture

```rust
// Extensible distance metric system
pub trait DistanceMetric: Send + Sync {
    fn compute(&self, a: &[f64], b: &[f64]) -> f64;
    fn name(&self) -> &'static str;
    fn supports_hnsw(&self) -> bool;
}

// Registration system for custom metrics
pub struct MetricRegistry {
    metrics: HashMap<String, Box<dyn DistanceMetric>>,
}

impl MetricRegistry {
    pub fn register_custom<T: DistanceMetric + 'static>(&mut self, metric: T) {
        self.metrics.insert(metric.name().to_string(), Box::new(metric));
    }
}
```

### 10.2 Algorithm Variants

```rust
// Framework for alternative algorithms
pub trait DimensionalityReduction {
    type Config;
    type Model;

    fn fit(&self, data: &Array2<f64>, config: Self::Config) -> Result<Self::Model, Error>;
    fn transform(&self, model: &Self::Model, data: &Array2<f64>) -> Result<Array2<f64>, Error>;
}

// PacMAP implementation
pub struct PacMAPAlgorithm;

impl DimensionalityReduction for PacMAPAlgorithm {
    type Config = PacMAPConfig;
    type Model = PacMAPModel;

    fn fit(&self, data: &Array2<f64>, config: Self::Config) -> Result<Self::Model, Error> {
        // Implementation
    }
}
```

---

**Document Version**: 1.0
**Last Updated**: September 2024
**Status**: Design Complete - Ready for Implementation