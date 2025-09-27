@echo off
setlocal enabledelayedexpansion
echo ===========================================
echo   PacMAP Enhanced Cross-Platform Build
echo ===========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker not available
    pause
    exit /b 1
)

REM Check Rust
rustc --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Rust not available! Please install Rust from https://rustup.rs/
    pause
    exit /b 1
)

REM === BUILD WINDOWS VERSION ===
echo [1/2] Building PacMAP Enhanced Windows version...
echo ----------------------------------------

echo [INFO] Building Windows release...
cargo build --release
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Windows build failed!
    exit /b 1
)

echo [INFO] Running Windows validation tests...
set WIN_ALL_TESTS_PASSED=1

echo [TEST] Running standard comprehensive validation...
cargo test test_standard_comprehensive --release -- --nocapture
if !ERRORLEVEL! EQU 0 (
    echo [PASS] Standard comprehensive test PASSED
) else (
    echo [FAIL] Standard comprehensive test FAILED with code !ERRORLEVEL!
    set WIN_ALL_TESTS_PASSED=0
)

echo [TEST] Running enhanced wrapper validation...
cargo test test_enhanced_wrapper --release -- --nocapture
if !ERRORLEVEL! EQU 0 (
    echo [PASS] Enhanced wrapper test PASSED
) else (
    echo [FAIL] Enhanced wrapper test FAILED with code !ERRORLEVEL!
    set WIN_ALL_TESTS_PASSED=0
)

echo [TEST] Running comprehensive pipeline validation...
cargo test test_comprehensive_pipeline --release -- --nocapture
if !ERRORLEVEL! EQU 0 (
    echo [PASS] Comprehensive pipeline test PASSED
) else (
    echo [FAIL] Comprehensive pipeline test FAILED with code !ERRORLEVEL!
    set WIN_ALL_TESTS_PASSED=0
)

echo [TEST] Running error fixes validation...
cargo test test_error_fixes_simple --release -- --nocapture
if !ERRORLEVEL! EQU 0 (
    echo [PASS] Error fixes test PASSED
) else (
    echo [FAIL] Error fixes test FAILED with code !ERRORLEVEL!
    set WIN_ALL_TESTS_PASSED=0
)

echo [TEST] Running metric validation...
cargo test test_metric_validation --release -- --nocapture
if !ERRORLEVEL! EQU 0 (
    echo [PASS] Metric validation test PASSED
) else (
    echo [FAIL] Metric validation test FAILED with code !ERRORLEVEL!
    set WIN_ALL_TESTS_PASSED=0
)

echo [TEST] Running quantization comprehensive validation...
cargo test test_quantization_comprehensive --release -- --nocapture
if !ERRORLEVEL! EQU 0 (
    echo [PASS] Quantization comprehensive test PASSED
) else (
    echo [FAIL] Quantization comprehensive test FAILED with code !ERRORLEVEL!
    set WIN_ALL_TESTS_PASSED=0
)

if !WIN_ALL_TESTS_PASSED! EQU 1 (
    echo [PASS] ALL Windows tests completed successfully
) else (
    echo [FAIL] Some Windows tests failed - build may have issues!
)

echo [PASS] Windows build completed

REM === BUILD LINUX VERSION WITH DOCKER ===
echo.
echo [2/2] Building PacMAP Enhanced Linux version with Docker...
echo ----------------------------------------

if exist build-linux (
    rmdir /s /q build-linux
)

REM Create Dockerfile for Rust cross-compilation
echo Creating temporary Dockerfile...
echo FROM ubuntu:22.04 > Dockerfile.temp
echo RUN apt-get update ^&^& apt-get install -y curl build-essential pkg-config libssl-dev >> Dockerfile.temp
echo RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs ^| sh -s -- -y >> Dockerfile.temp
echo ENV PATH="/root/.cargo/bin:${PATH}" >> Dockerfile.temp
echo WORKDIR /src >> Dockerfile.temp
echo COPY . . >> Dockerfile.temp
echo RUN cargo build --release >> Dockerfile.temp
echo RUN cargo test --release --no-run >> Dockerfile.temp

REM Build Docker image and run build
echo [INFO] Building Docker image and compiling for Linux...
docker build -f Dockerfile.temp -t pacmap-enhanced-linux .
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker Linux build failed!
    del Dockerfile.temp
    exit /b 1
)

REM Extract built artifacts
echo [INFO] Extracting Linux build artifacts...
docker create --name pacmap-temp pacmap-enhanced-linux
docker cp pacmap-temp:/src/target/release/libpacmap_enhanced.so build-linux/libpacmap_enhanced.so 2>nul
docker cp pacmap-temp:/src/target/release/libpacmap_enhanced.a build-linux/libpacmap_enhanced.a 2>nul
docker cp pacmap-temp:/src/target/release/pacmap-enhanced build-linux/pacmap-enhanced 2>nul
docker rm pacmap-temp

REM Clean up
del Dockerfile.temp

if exist build-linux\libpacmap_enhanced.so (
    echo [PASS] Linux shared library created successfully
) else (
    echo [INFO] Linux shared library not found (may not be configured for C FFI export)
)

echo [PASS] Linux build completed

REM === SETUP CSHARP PROJECT STRUCTURE ===
echo.
echo [3/3] Setting up PacMAP Enhanced Sharp project...
echo ----------------------------------------

REM Create C# project structure
if not exist "..\PacMAPSharp" (
    echo [INFO] Creating PacMAPSharp project structure
    mkdir "..\PacMAPSharp"
)

if not exist "..\PacMAPSharp\PacMAPSharp" (
    mkdir "..\PacMAPSharp\PacMAPSharp"
)

REM Copy Windows DLL if it exists (would need to be built as cdylib)
if exist "target\release\pacmap_enhanced.dll" (
    copy "target\release\pacmap_enhanced.dll" "..\PacMAPSharp\PacMAPSharp\pacmap_enhanced.dll"
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Copied Windows pacmap_enhanced.dll to C# project
    ) else (
        echo [FAIL] Failed to copy Windows DLL - Error: !ERRORLEVEL!
    )
) else (
    echo [INFO] Windows DLL not found (Rust project may need cdylib configuration)
)

REM Copy Linux .so file if it exists
if exist "build-linux\libpacmap_enhanced.so" (
    copy "build-linux\libpacmap_enhanced.so" "..\PacMAPSharp\PacMAPSharp\libpacmap_enhanced.so"
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Copied Linux libpacmap_enhanced.so to C# project
    ) else (
        echo [FAIL] Failed to copy Linux library - Error: !ERRORLEVEL!
    )
) else (
    echo [INFO] Linux .so file not found in build-linux directory
)

echo [PASS] PacMAP Enhanced libraries setup completed

REM === SUMMARY ===
echo.
echo ===========================================
echo   PacMAP Enhanced Build Summary
echo ===========================================
echo.
echo Windows libraries (target\release\):
if exist "target\release\pacmap_enhanced.exe" (
    echo   [PASS] pacmap_enhanced.exe (Main executable)
    for %%A in ("target\release\pacmap_enhanced.exe") do echo         Size: %%~zA bytes
) else (echo   [INFO] pacmap_enhanced.exe (executable may not be configured))

if exist "target\release\pacmap_enhanced.dll" (
    echo   [PASS] pacmap_enhanced.dll (FFI library for C#)
    for %%A in ("target\release\pacmap_enhanced.dll") do echo         Size: %%~zA bytes
) else (echo   [INFO] pacmap_enhanced.dll (may need cdylib configuration in Cargo.toml))

echo.
echo Linux libraries (build-linux\):
if exist "build-linux" (
    dir build-linux\libpacmap_enhanced.so* build-linux\*.so /B 2>nul | findstr /R ".*"
    if !ERRORLEVEL! EQU 0 (
        echo   [PASS] Linux .so files found:
        for %%F in (build-linux\libpacmap_enhanced.so* build-linux\*.so) do (
            if exist "%%F" (
                for %%A in ("%%F") do echo         %%~nxF (%%~zA bytes)
            )
        )
    ) else (
        echo   [INFO] No Linux .so files found (may need cdylib configuration)
    )
) else (
    echo   [INFO] build-linux directory not found
)

echo.
echo PacMAPSharp C# project files:
if exist "..\PacMAPSharp\PacMAPSharp\pacmap_enhanced.dll" (
    echo   [PASS] Windows library: pacmap_enhanced.dll
    for %%A in ("..\PacMAPSharp\PacMAPSharp\pacmap_enhanced.dll") do echo         Size: %%~zA bytes
) else (echo   [INFO] Windows library not copied (needs cdylib build type))

if exist "..\PacMAPSharp\PacMAPSharp\libpacmap_enhanced.so" (
    echo   [PASS] Linux library: libpacmap_enhanced.so
    for %%A in ("..\PacMAPSharp\PacMAPSharp\libpacmap_enhanced.so") do echo         Size: %%~zA bytes
) else (echo   [INFO] Linux library not copied (needs cdylib build type))

echo.
echo ===========================================
echo   PacMAP Enhanced Features Available
echo ===========================================
echo.
echo Your PacMAP Enhanced library now supports:
echo   - HNSW-optimized neighbor search for large datasets
echo   - Advanced quantization with parameter preservation
echo   - Multiple normalization modes (ZScore, MinMax, Robust)
echo   - Comprehensive model persistence with MessagePack + ZSTD compression
echo   - Enhanced FFI wrapper for cross-language integration
echo   - Auto-scaling HNSW parameters based on dataset characteristics
echo   - Cross-platform support (Windows + Linux)
echo   - Comprehensive test suite with 6 validation categories
echo.
echo ===========================================
echo   Ready for Development!
echo ===========================================
echo.
echo Next steps:
if exist "..\PacMAPSharp\PacMAPSharp\pacmap_enhanced.dll" (
    echo   1. Build C# library: dotnet build ..\PacMAPSharp\PacMAPSharp
    echo   2. Create NuGet package: dotnet pack ..\PacMAPSharp\PacMAPSharp --configuration Release
    echo   3. Test advanced features: model.FitTransformNormalized(data, normalizationMode);
) else (
    echo   1. Configure Cargo.toml for C FFI: Add [lib] crate-type = ["cdylib"]
    echo   2. Rebuild with: cargo build --release
    echo   3. Then build C# wrapper project
)
echo   4. Run comprehensive tests: cargo test --release
echo   5. Test quantization: model.QuantizeOnSave = true; model.SaveCompressed(path);
echo.
pause