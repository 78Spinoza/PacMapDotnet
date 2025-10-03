# PacMAP Enhanced - Build Instructions

## Overview

PacMAP Enhanced is a comprehensive dimensionality reduction library built in Rust with advanced features including HNSW neighbor search, quantization, progress reporting, and cross-platform support.

## Features

- 🔍 **HNSW-optimized neighbor search** for large datasets
- 🗜️ **Advanced quantization** with parameter preservation
- 📊 **Multiple normalization modes** (ZScore, MinMax, Robust)
- 💾 **Model persistence** with MessagePack + ZSTD compression
- 🔄 **Real-time progress reporting** with detailed phase tracking
- 🎯 **Deterministic seed preservation** for reproducible results
- 🌐 **Cross-platform support** (Windows + Linux)
- ✅ **Comprehensive test suite** (6 validation categories)

## Prerequisites

### Windows
- **Rust**: Install from [https://rustup.rs/](https://rustup.rs/)
- **Visual Studio Build Tools**: Required for compilation
- **Docker Desktop**: For Linux cross-compilation (optional)

### Linux
- **Rust**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Build essentials**: `apt-get install build-essential pkg-config libssl-dev`

## Quick Start

### 1. Clone and Build
```bash
git clone <repository-url>
cd pacmap-enhanced
cargo build --release
```

### 2. Run Comprehensive Tests
```bash
# Windows - All 6 test suites
./build_test.bat

# Linux/Manual
cargo test --release -- --nocapture
```

## Build Options

### Option 1: Windows Testing (Recommended)
Run the comprehensive test suite:
```cmd
build_test.bat
```
This executes all 6 test categories in debug mode (to avoid DLL issues):
- Standard comprehensive validation
- Enhanced wrapper validation
- Comprehensive pipeline validation
- Error fixes validation
- Metric validation
- Quantization comprehensive validation

**Note**: The BAT file uses debug builds to avoid Windows DLL dependency issues with release mode.

### Option 2: Cross-Platform Build
Build for both Windows and Linux using Docker:
```cmd
BuildDockerLinuxWindows.bat
```
This will:
- Build and test Windows version
- Create Linux build using Docker
- Set up C# project structure (if applicable)
- Copy libraries for deployment

### Option 3: Manual Build
```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run specific test
cargo test test_progress_callback -- --nocapture
```

## Test Suite Details

### 1. Standard Comprehensive (`test_standard_comprehensive`)
- Tests 2D and 10D embeddings
- MSE validation between fit and transform
- Error rate calculation (max 0.5% points with >1% error)
- Coordinate variety validation

### 2. Enhanced Wrapper (`test_enhanced_wrapper`)
- FFI (Foreign Function Interface) testing
- Progress callback validation
- Version info and configuration testing
- Model persistence through C interface

### 3. Comprehensive Pipeline (`test_comprehensive_pipeline`)
- End-to-end pipeline validation
- Multiple normalization modes
- k-NN method comparison (brute-force vs HNSW)
- Multi-dimensional embedding testing (1D to 20D)

### 4. Error Fixes Simple (`test_error_fixes_simple`)
- Basic training with normalization
- Save/load validation
- HNSW parameter validation
- Memory management testing
- Edge case handling

### 5. Metric Validation (`test_metric_validation`)
- Euclidean distance metric validation
- Normalization consistency testing
- Constant data handling
- High-dimensional data validation
- Sparse data processing

### 6. Quantization Comprehensive (`test_quantization_comprehensive`)
- Quantization parameter preservation
- Compression ratio testing
- Error rate statistics
- Save/load consistency validation
- Performance metrics collection

## Progress Reporting

PacMAP Enhanced includes comprehensive progress reporting:

### Basic Usage
```rust
use pacmap_enhanced::{fit_transform_normalized_with_progress, ProgressCallback};

let progress_callback: ProgressCallback = Box::new(|phase, current, total, percent, message| {
    println!("[{:>12}] {:>3}% ({:>3}/{:<3}) - {}",
             phase, percent as u32, current, total, message);
});

let (embedding, model) = fit_transform_normalized_with_progress(
    data, config, normalization_mode, Some(progress_callback)
)?;
```

### Progress Phases
1. **Initializing** (0%) - Preparing dataset
2. **Analyzing** (5%) - Data characteristics analysis
3. **Normalizing** (10%) - Applying normalization
4. **HNSW Config** (20%) - Auto-scaling HNSW parameters
5. **HNSW Ready** (25%) - HNSW configuration complete
6. **Embedding** (30%) - Computing PacMAP embedding
7. **Embedding Done** (80%) - Embedding computation complete
8. **Finalizing** (90%) - Computing statistics
9. **Complete** (100%) - Process finished

### Model Settings Display
The system automatically displays comprehensive model settings:

```
🎛️  Core PacMAP Configuration:
   - Embedding dimensions: 2D
   - Number of neighbors: 15
   - Random seed: 42 (deterministic)

🔍 HNSW Neighbor Search Configuration:
   - M parameter: 16
   - ef_construction: 64
   - ef_search: 114

📊 Data Normalization Configuration:
   - Mode: ZScore
   - Features: 10

🎯 Embedding Information:
   - Shape: [200, 2]
   - Non-zero values: 400/400 (100.0%)

📏 Distance Statistics:
   - Mean distance: 18.032939
   - 95th percentile: 38.068863
```

## API Usage

### Basic Fitting
```rust
use pacmap_enhanced::{fit_transform_normalized, stats::NormalizationMode};
use ndarray::Array2;

let data: Array2<f64> = /* your data */;
let config = pacmap::Configuration {
    embedding_dimensions: 2,
    override_neighbors: Some(15),
    seed: Some(42),
    ..Default::default()
};

let (embedding, model) = fit_transform_normalized(
    data,
    config,
    Some(NormalizationMode::ZScore)
)?;
```

### Model Persistence
```rust
// Save with progress reporting
model.save_compressed_with_progress("model.bin", Some(progress_callback))?;

// Load with progress reporting
let loaded_model = PaCMAP::load_compressed_with_progress("model.bin", Some(progress_callback))?;
```

### Transform New Data
```rust
use pacmap_enhanced::transform_with_model;

let new_embedding = transform_with_model(&model, new_data)?;
```

## Configuration Options

### HNSW Parameters (Auto-scaled)
```rust
// Parameters are automatically scaled based on dataset size
// Manual override available through HnswParams::custom()
```

### Normalization Modes
- `NormalizationMode::ZScore` - Standard standardization
- `NormalizationMode::MinMax` - Min-max scaling [0,1]
- `NormalizationMode::Robust` - Robust scaling using median/IQR
- `NormalizationMode::None` - No normalization

### Quantization
```rust
// Enable quantization for smaller file sizes
model.quantize_on_save = true;
model.save_compressed("quantized_model.bin")?;
```

## Troubleshooting

### Common Issues

1. **Release Build DLL Issues on Windows**
   - Use debug builds: `cargo build` instead of `cargo build --release`
   - Or temporarily disable real-time antivirus protection
   - Add project directory to antivirus exclusions

2. **Docker Not Available**
   - Install Docker Desktop for Windows
   - Ensure Docker service is running

3. **OpenBLAS Errors**
   - Check C:\PacMAN\lapack-binaries exists
   - Verify build.rs configuration

4. **Memory Issues with Large Datasets**
   - Use HNSW auto-scaling (automatic)
   - Consider data subsampling for initial testing

### Performance Tips

1. **For Large Datasets (>10k samples)**
   - HNSW will automatically activate
   - Consider using fewer neighbors initially
   - Monitor progress with callbacks

2. **For Production Use**
   - Always use release builds (`--release`)
   - Enable progress callbacks for monitoring
   - Use quantization for model storage optimization

3. **For Reproducible Results**
   - Always set a seed in configuration
   - Use identical normalization modes
   - Preserve the exact model file

## Dependencies

Key dependencies automatically managed by Cargo:
- `pacmap = "0.2"` - Core PacMAP algorithm
- `ndarray = "0.16"` - N-dimensional arrays
- `serde = "1.0"` - Serialization framework
- `rmp-serde = "0.15"` - MessagePack serialization
- `zstd = "0.13"` - Compression
- `half = "2.4"` - f16 quantization support
- `rand = "0.8"` - Random number generation (dev-dependencies)

## FFI Support

For C# integration, the library provides:
- C-compatible FFI functions
- Progress callback support
- Model persistence through C interface
- Cross-platform library generation

## Cross-Platform Deployment

The BuildDockerLinuxWindows.bat script creates:
- `pacmap_enhanced.dll` (Windows)
- `libpacmap_enhanced.so` (Linux)

Ready for deployment in:
- C# applications (.NET Core/Framework)
- Python bindings
- Native C/C++ applications

## Support

For issues and questions:
1. Check this build documentation
2. Review test suite outputs
3. Enable progress callbacks for debugging
4. Check GitHub issues (if available)

---

## Build Results Validation

After building, you should see:
- ✅ All 6 test suites passing
- ✅ Progress reporting working
- ✅ Model settings displayed correctly
- ✅ Cross-platform libraries generated
- ✅ Zero compilation warnings

**PacMAP Enhanced is ready for production use!** 🚀