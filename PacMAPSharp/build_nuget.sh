#!/bin/bash

echo "============================================"
echo "Building PacMAPSharp NuGet Package - Linux"
echo "============================================"

# Clean previous builds
echo "[1/5] Cleaning previous builds..."
if [ -d "bin" ]; then rm -rf bin; fi
if [ -d "obj" ]; then rm -rf obj; fi
if [ -d "nupkg" ]; then rm -rf nupkg; fi

# Ensure LAPACK binaries are available
echo "[2/5] Setting up LAPACK binaries..."
if [ ! -f "libopenblas.so.0" ]; then
    echo "ERROR: LAPACK binaries not found. Please download from README.md"
    exit 1
fi

# Build Rust core library first (if source available)
echo "[3/5] Building Rust core library..."
if [ -d "../pacmap-enhanced" ]; then
    cd ../pacmap-enhanced
    export OPENBLAS_PATH=../lapack-binaries/linux
    cargo build --release
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build Rust library"
        cd ../PacMAPSharp
        exit 1
    fi
    cd ../PacMAPSharp

    # Copy latest native binaries
    echo "[4/5] Copying native binaries..."
    cp "../pacmap-enhanced/target/release/pacmap_enhanced.dll" "pacmap_enhanced.dll" 2>/dev/null || cp "pacmap_enhanced.dll" "pacmap_enhanced.dll"
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed to copy Rust DLL, using existing binary"
    fi
else
    echo "[4/5] Using existing native binaries..."
fi

# Copy OpenBLAS libraries
cp "libopenblas.so.0" "libopenblas.so.0"
if [ ! -f "libopenblas.so.0" ]; then
    echo "ERROR: Failed to copy OpenBLAS library"
    exit 1
fi

# Build C# library
echo "[5/5] Building C# library..."
dotnet build --configuration Release
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build C# library"
    exit 1
fi

# Pack NuGet
echo "[6/6] Creating NuGet package..."
dotnet pack --configuration Release --no-build --output nupkg
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create NuGet package"
    exit 1
fi

echo
echo "============================================"
echo "SUCCESS: PacMAPSharp NuGet package built!"
echo "============================================"
echo "Package location: nupkg/"
ls nupkg/*.nupkg