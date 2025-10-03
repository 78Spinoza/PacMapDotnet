# LAPACK Binaries for PacMAP Build

This directory contains LAPACK/OpenBLAS binaries needed for building PacMAP on different platforms.

## Directory Structure

- `windows/` - Windows x64 OpenBLAS binaries (libopenblas.dll)
- `linux/` - Linux x64 OpenBLAS libraries (libopenblas.so.0)

## Build Instructions

### For Windows
1. Copy `windows/libopenblas.dll` to your build directory
2. Add `windows/` directory to your PATH or LIBRARY_PATH
3. Build with: `cargo build --release`

### For Linux
1. Copy `linux/libopenblas.so.0` to your build directory or `/usr/local/lib/`
2. Run: `sudo ldconfig` (if copied to system directory)
3. Build with: `cargo build --release`

### For C# Wrapper
The build scripts automatically copy the appropriate binaries:
- Windows: `libopenblas.dll` → `PacMAPSharp/libopenblas.dll`
- Linux: `libopenblas.so.0` → `PacMAPSharp/libopenblas.so.0`

## Source
Downloaded from: https://github.com/OpenMathLib/OpenBLAS/releases
Version: v0.3.28

## Notes
- These binaries are included to save build time for other developers
- If you need different architectures, download from the source above
- Windows: x64 architecture
- Linux: x86_64 manylinux1 compatible