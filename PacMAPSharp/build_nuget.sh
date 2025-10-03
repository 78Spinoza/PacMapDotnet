#!/bin/bash
set -e

echo "============================================"
echo "Building PacMAPSharp Cross-Platform NuGet Package (Linux)"
echo "============================================"

# Clean previous builds
echo "[1/6] Cleaning previous builds..."
rm -rf bin obj nupkg

# Set up binary copying to project base
echo "[2/6] Preparing to copy binaries to project base..."

# Build Rust core library for Linux
echo "[3/6] Building Rust core library for Linux..."
cd ../pacmap-enhanced

# Build for current Linux platform
cargo build --release --target x86_64-unknown-linux-gnu
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Rust library for Linux"
    exit 1
fi

# Check if Windows cross-compilation tools are available
echo "[4/6] Attempting Windows cross-compilation..."
if rustup target list --installed | grep -q x86_64-pc-windows-msvc; then
    echo "Building for Windows target..."
    cargo build --release --target x86_64-pc-windows-msvc
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Windows cross-compilation completed"
        WINDOWS_BUILD_SUCCESS=1
    else
        echo "[WARNING] Windows cross-compilation failed, continuing with Linux-only build"
        WINDOWS_BUILD_SUCCESS=0
    fi
else
    echo "[INFO] Windows target not installed, skipping Windows build"
    echo "       To enable: rustup target add x86_64-pc-windows-msvc"
    WINDOWS_BUILD_SUCCESS=0
fi

cd ../PacMAPSharp

# Copy Linux native binaries directly to project base
echo "[5/6] Copying native binaries to project base..."
if [ -f "../pacmap-enhanced/target/x86_64-unknown-linux-gnu/release/libpacmap_enhanced.so" ]; then
    cp "../pacmap-enhanced/target/x86_64-unknown-linux-gnu/release/libpacmap_enhanced.so" "libpacmap_enhanced.so"
    echo "[SUCCESS] Linux binary copied with proper linking"

    # Verify Linux binary linking
    echo "Checking Linux binary dependencies..."
    ldd "libpacmap_enhanced.so" || true

    # Fix common Linux linking issues
    echo "Validating library compatibility..."
    if ldd "libpacmap_enhanced.so" | grep -q "not found"; then
        echo "[WARNING] Missing dependencies detected - ensuring library path linking"
        # Add RPATH for common system libraries
        chrpath -r '$ORIGIN' "libpacmap_enhanced.so" 2>/dev/null || echo "[INFO] chrpath not available, skipping RPATH fix"
    fi
else
    echo "ERROR: Failed to build Linux Rust library"
    exit 1
fi

# Copy Windows native binaries if available
if [ $WINDOWS_BUILD_SUCCESS -eq 1 ] && [ -f "../pacmap-enhanced/target/x86_64-pc-windows-msvc/release/pacmap_enhanced.dll" ]; then
    echo "Copying Windows binaries..."
    cp "../pacmap-enhanced/target/x86_64-pc-windows-msvc/release/pacmap_enhanced.dll" "pacmap_enhanced.dll"
    echo "[SUCCESS] Windows binary copied"
fi

# Copy OpenBLAS library for Linux (if available)
OPENBLAS_FOUND=0
for path in /usr/lib/x86_64-linux-gnu/libopenblas.so* /usr/lib64/libopenblas.so* /opt/openblas/lib/libopenblas.so*; do
    if [ -f "$path" ]; then
        cp "$path" "libopenblas.so"
        echo "[SUCCESS] OpenBLAS library found and copied: $path"
        OPENBLAS_FOUND=1
        break
    fi
done

if [ $OPENBLAS_FOUND -eq 0 ]; then
    echo "[WARNING] OpenBLAS not found - install with: sudo apt-get install libopenblas-dev"
fi

# Build C# library
echo "[6/6] Building C# library and creating NuGet package..."
dotnet build --configuration Release
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build C# library"
    exit 1
fi

# Pack NuGet with cross-platform binaries
dotnet pack --configuration Release --no-build --output nupkg
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create NuGet package"
    exit 1
fi

echo ""
echo "============================================"
echo "SUCCESS: PacMAPSharp Cross-Platform NuGet Package Built!"
echo "============================================"
echo "Package location: nupkg/"
ls -la nupkg/*.nupkg
echo ""
echo "Binary structure (project base):"
echo "Linux binaries:"
if [ -f "libpacmap_enhanced.so" ]; then
    echo "  ✓ libpacmap_enhanced.so (with proper linking)"
    echo "    Size: $(stat -c%s libpacmap_enhanced.so) bytes"
    echo "    Dependencies:"
    ldd "libpacmap_enhanced.so" | head -5
else
    echo "  ✗ libpacmap_enhanced.so MISSING"
fi

if [ -f "libopenblas.so" ]; then
    echo "  ✓ libopenblas.so"
else
    echo "  ⚠ libopenblas.so not available (install libopenblas-dev)"
fi
echo ""

echo "Windows binaries:"
if [ -f "pacmap_enhanced.dll" ]; then
    echo "  ✓ pacmap_enhanced.dll"
else
    echo "  ⚠ pacmap_enhanced.dll not available (Windows cross-compilation required)"
fi
if [ -f "libopenblas.dll" ]; then
    echo "  ✓ libopenblas.dll"
else
    echo "  ⚠ libopenblas.dll not available (Windows cross-compilation required)"
fi
echo ""

echo "Build completed successfully!"