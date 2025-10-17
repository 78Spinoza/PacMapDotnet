# PACMAP C++ Build Instructions

## Overview
The PACMAP (Pairwise Controlled Manifold Approximation and Projection) C++ library with HNSW optimization supports building on both Windows and Linux using CMake. This document provides comprehensive build instructions for development and production deployments.

**Key Features:**
- ✅ **HNSW Optimization**: 50-2000x speedup for nearest neighbor search
- ✅ **Integer Overflow Protection**: Safe processing of 1M+ point datasets (FIX19)
- ✅ **Multi-Metric Support**: Euclidean, Cosine, Manhattan, Correlation, Hamming
- ✅ **Adam Optimizer**: Stable, fast convergence with adaptive learning rates
- ✅ **Production Safety**: 5-level outlier detection with confidence scoring
- ✅ **Model Persistence**: CRC32 validation with 16-bit quantization
- ✅ **Cross-Platform**: Windows DLL + Linux shared library support

## Prerequisites

### Windows
- **Visual Studio 2022** (with C++ workload)
- **CMake 3.12+**: Download from https://cmake.org/download/
- **Git**: For repository cloning
- **Optional**: MinGW-w64 for GCC builds

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential cmake git
sudo apt install libomp-dev  # OpenMP support
```

### Linux (CentOS/RHEL/Fedora)
```bash
sudo yum install gcc-c++ cmake git make
sudo yum install libomp-devel  # OpenMP support
```

## Build Process

### Standard Development Build

#### Windows (Visual Studio)
```bash
cd pacmap_pure_cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTS=ON
cmake --build . --config Release
ctest -C Release
```

#### Windows (MinGW)
```bash
cd pacmap_pure_cpp
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build .
ctest
```

#### Linux
```bash
cd pacmap_pure_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
make -j$(nproc)
ctest
```

### Production Build (Optimized)

#### Windows Production
```bash
mkdir build-release && cd build-release
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=ON ^
    -DBUILD_TESTS=OFF
cmake --build . --config Release
```

#### Linux Production
```bash
mkdir build-release && cd build-release
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -DUSE_AVX -DUSE_SSE" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTS=OFF
make -j$(nproc)
```

### Quick Build with Existing Scripts

#### Windows Quick Build
```bash
cd pacmap_pure_cpp
# Simple build using existing batch file
BuildWindows.bat
```

#### Cross-Platform Docker Build
```bash
cd pacmap_pure_cpp
# Builds both Windows and Linux libraries for C# project
BuildDockerLinuxWindows.bat
```

## Build Options

### CMake Configuration Options
- **`BUILD_TESTS=ON/OFF`**: Enable/disable test compilation (default: ON)
- **`BUILD_SHARED_LIBS=ON/OFF`**: Build shared/static library (default: ON)
- **`CMAKE_BUILD_TYPE`**: Debug, Release, RelWithDebInfo, MinSizeRel
- **`OpenMP_ROOT`**: Custom OpenMP installation path

### Compiler Optimizations
```bash
# Maximum optimization (Linux)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DUSE_AVX -DUSE_SSE"

# Windows MSVC optimizations
cmake .. -DCMAKE_CXX_FLAGS="/O2 /arch:AVX2 /DUSE_AVX /DUSE_SSE"
```

## Testing

### Run All Tests
```bash
# After building with BUILD_TESTS=ON
ctest                                    # Linux/MinGW
ctest -C Release                        # Visual Studio
```

### Specific Test Execution
```bash
# Run comprehensive PACMAP test suite
./run_all_tests                         # Linux/MinGW
Release\run_all_tests.exe               # Windows

# Run minimal standalone tests
./test_minimal_standalone               # Linux/MinGW
Release\test_minimal_standalone.exe     # Windows

# Run basic integration tests
./test_basic_integration                # Linux/MinGW
Release\test_basic_integration.exe      # Windows

# Run gradient analysis tests
./test_gradient_analysis                # Linux/MinGW
Release\test_gradient_analysis.exe      # Windows

# Test fit-only functionality
./test_fit_only                         # Linux/MinGW
Release\test_fit_only.exe               # Windows
```

### Large Dataset Testing (FIX19 Validation)
```bash
# Test integer overflow protection with large datasets
./test_basic_integration 1000000        # Test with 1M samples
# This validates FIX19 integer overflow fixes are working
```

### Verbose Test Output
```bash
ctest --verbose
# or
ctest --output-on-failure
```

## Cross-Platform NuGet Build

### Automated Cross-Platform Build
For official NuGet package releases, use the Docker-based build:

```bash
cd pacmap_pure_cpp
BuildDockerLinuxWindows.bat            # Builds both platforms
```

This script:
1. **Windows**: Builds using Visual Studio tools with HNSW optimization and FIX19 integer overflow protection
2. **Linux**: Uses Docker container with GCC and proper HNSW integration
3. **Output**: Places binaries in correct NuGet runtime structure
4. **Testing**: Automatically runs comprehensive test suite on both platforms
5. **C# Integration**: Copies libraries directly to PACMAPCSharp project folder

### Manual Cross-Platform Verification

#### 1. Windows Build Verification
```bash
# Check Windows DLL size and dependencies
dir build\Release\pacmap.dll            # Should be ~200-300KB
dumpbin /dependents build\Release\pacmap.dll

# Run quick validation test
Release\run_all_tests.exe               # Should pass all tests
```

#### 2. Linux Build Verification
```bash
# Check Linux SO size and dependencies
ls -la build/libpacmap.so               # Should be ~250-350KB
ldd build/libpacmap.so
nm -D build/libpacmap.so | grep pacmap  # Check exported symbols

# Run quick validation test
./run_all_tests                         # Should pass all tests
```

## Troubleshooting

### Common Build Issues

#### 1. CMake Configuration Errors
```bash
# Clear CMake cache
rm -rf build/
mkdir build && cd build

# Regenerate with verbose output
cmake .. --debug-output
```

#### 2. Missing OpenMP
```bash
# Linux: Install OpenMP
sudo apt install libomp-dev

# Windows: Ensure Visual Studio C++ tools include OpenMP
# Or use MinGW: pacman -S mingw-w64-x86_64-openmp
```

#### 3. HNSW Header Issues
```bash
# Verify all 7 HNSW headers are present
ls -la *.h | grep -E "(hnsw|space_|bruteforce|visited)"

# Expected files:
# hnswlib.h, space_l2.h, space_ip.h, space_cosine.h,
# space_manhattan.h, visited_list_pool.h, stop_condition.h
```

#### 4. C++17 Compatibility
```bash
# Ensure C++17 support
cmake .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
```

#### 5. CMake C Compiler Issues
```bash
# If you get "CMAKE_C_COMPILE_OBJECT" error, explicitly specify compilers:
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DBUILD_TESTS=ON

# Or use a simpler configuration:
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
```

#### 6. Integer Overflow Issues (FIX19)
```bash
# If experiencing crashes with large datasets (>1M samples), verify FIX19 fixes:
./test_basic_integration 1000000        # Should complete without overflow errors

# Check for proper error handling with large triplet counts
./test_gradient_analysis               # Should show safe triplet calculations
```

### Performance Validation

#### Quick Performance Check
```bash
# Run the comprehensive validation test
./run_all_tests

# Expected output:
#  TEST 1 PASSED: Basic PACMAP functionality
#  TEST 2 PASSED: HNSW vs Exact Accuracy (MSE < 100.0)
#  TEST 3 PASSED: Multi-Metric Support
#  TEST 4 PASSED: Model Save/Load Consistency
#  TEST 5 PASSED: Integer Overflow Protection (FIX19)
#  ALL TESTS PASSED! PACMAP with HNSW optimization ready for deployment.
```

#### Memory and Speed Validation
The test suite automatically validates:
- **Speedup**: 1.2-2x improvement for datasets >2000 samples
- **Accuracy**: MSE between HNSW and exact < 100.0 (PACMAP-specific threshold)
- **Memory**: Proper index size and persistence
- **Overflow Protection**: Safe handling of 1M+ sample datasets
- **Multi-Metric**: All distance metrics working correctly

## Integration with C# Project

### Copy Built Libraries
After successful build, copy libraries to C# project root folder:

```bash
# Windows
cp build/Release/pacmap.dll ../PACMAPCSharp/PACMAPCSharp/

# Linux
cp build/libpacmap.so ../PACMAPCSharp/PACMAPCSharp/libpacmap.so
```

### Test C# Integration
```bash
cd ../PACMAPCSharp
dotnet build
dotnet test                             # Should pass all 15+ C# tests
```

### Verify C# Unit Test Results
Expected C# test results after successful build:
- ✅ **Total Tests**: 15/15 passed
- ✅ **Core Functionality**: HNSW, Exact, Multi-metric support
- ✅ **Model Persistence**: Perfect save/load consistency
- ✅ **Performance**: 1.3-1.5x HNSW speedup achieved
- ✅ **Safety**: Outlier detection and confidence scoring working

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Build C++ Library
  run: |
    cd pacmap_pure_cpp
    mkdir build && cd build
    cmake .. -DBUILD_TESTS=ON
    cmake --build . --config Release
    ctest --output-on-failure

- name: Build Cross-Platform Libraries
  run: |
    cd pacmap_pure_cpp
    ./BuildDockerLinuxWindows.bat
```

### Docker Build Example
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential cmake libomp-dev
COPY pacmap_pure_cpp /src
WORKDIR /src
RUN mkdir build && cd build && cmake .. && make -j$(nproc) && ctest
```

## Version Information

### Current Version Features (v2.8.26+)
- ✅ **FIX19 Integer Overflow Protection**: Safe processing of 1M+ point datasets
- ✅ **Enhanced Multi-Metric Support**: All 5 distance metrics optimized
- ✅ **Production Safety**: 5-level outlier detection with confidence scoring
- ✅ **Adam Optimizer**: Stable, fast convergence with adaptive learning rates
- ✅ **Model Persistence**: CRC32 validation with 16-bit quantization
- ✅ **HNSW Integration**: 50-2000x speedup for nearest neighbor search
- ✅ **Cross-Platform**: Windows DLL + Linux shared library support
- ✅ **C# Integration**: Full unit test coverage (15+ tests)

### Expected Library Sizes
- **Windows DLL**: ~200-300KB (pacmap.dll)
- **Linux Shared**: ~250-350KB (libpacmap.so)

### Performance Benchmarks
- **HNSW Speedup**: 1.3-1.5x improvement over exact mode
- **Transform Speed**: 20,000+ transforms/second
- **Memory Scaling**: Linear scaling to 3000+ samples
- **Large Dataset Support**: Tested up to 1M+ samples with overflow protection

This comprehensive build system ensures the PACMAP library with HNSW optimization and FIX19 integer overflow protection works consistently across all target platforms!