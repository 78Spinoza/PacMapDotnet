@echo off
setlocal enabledelayedexpansion
echo ============================================
echo Building PacMAPSharp Cross-Platform NuGet Package
echo ============================================

rem Clean previous builds
echo [1/6] Cleaning previous builds...
if exist bin rmdir /s /q bin
if exist obj rmdir /s /q obj
if exist nupkg rmdir /s /q nupkg

rem Set up binary copying to project base
echo [2/6] Preparing to copy binaries to project base...

rem Build Rust core library for Windows
echo [3/6] Building Rust core library for Windows...
cd ..\pacmap-enhanced
cargo build --release
if errorlevel 1 (
    echo ERROR: Failed to build Rust library for Windows
    cd ..\PacMAPSharp
    exit /b 1
)

rem Build Rust core library for Linux using Docker
echo [4/6] Building Rust core library for Linux...
echo Creating Linux build environment...

rem Create temporary Dockerfile for Rust Linux compilation
echo FROM rust:1.75-slim > Dockerfile.linux
echo RUN apt-get update ^&^& apt-get install -y build-essential pkg-config libssl-dev >> Dockerfile.linux
echo RUN rustup target add x86_64-unknown-linux-gnu >> Dockerfile.linux
echo WORKDIR /build >> Dockerfile.linux
echo COPY . . >> Dockerfile.linux
echo RUN cargo build --release --target x86_64-unknown-linux-gnu >> Dockerfile.linux
echo CMD ["ls", "-la", "target/x86_64-unknown-linux-gnu/release/"] >> Dockerfile.linux

rem Build Linux version with Docker
docker build -f Dockerfile.linux -t pacmap-linux-build .
if errorlevel 1 (
    echo WARNING: Docker Linux build failed, continuing with Windows-only build
    del Dockerfile.linux
    cd ..\PacMAPSharp
    goto :skip_linux
) else (
    echo [SUCCESS] Linux build completed
)

rem Extract Linux binaries with proper linking
echo Extracting Linux binaries...
docker create --name pacmap-linux-temp pacmap-linux-build
docker cp pacmap-linux-temp:/build/target/x86_64-unknown-linux-gnu/release/libpacmap_enhanced.so ./target/linux/
docker rm pacmap-linux-temp
del Dockerfile.linux

:skip_linux
cd ..\PacMAPSharp

rem Copy Windows native binaries directly to project base
echo [5/6] Copying native binaries to project base...
copy "..\pacmap-enhanced\target\release\pacmap_enhanced.dll" "pacmap_enhanced.dll" /Y
if not exist "pacmap_enhanced.dll" (
    echo ERROR: Failed to copy Windows Rust DLL
    exit /b 1
)

copy "..\lapack-binaries\bin\libopenblas.dll" "libopenblas.dll" /Y
if not exist "libopenblas.dll" (
    echo ERROR: Failed to copy Windows OpenBLAS DLL
    exit /b 1
)

rem Copy Linux native binaries if available
if exist "..\pacmap-enhanced\target\linux\libpacmap_enhanced.so" (
    echo Copying Linux binaries...
    copy "..\pacmap-enhanced\target\linux\libpacmap_enhanced.so" "libpacmap_enhanced.so" /Y
    echo [SUCCESS] Linux binary copied with proper linking
) else (
    echo [WARNING] Linux binary not available - Windows-only build
)

rem Build C# library
echo [6/6] Building C# library and creating NuGet package...
dotnet build --configuration Release
if errorlevel 1 (
    echo ERROR: Failed to build C# library
    exit /b 1
)

rem Pack NuGet with cross-platform binaries
dotnet pack --configuration Release --no-build --output nupkg
if errorlevel 1 (
    echo ERROR: Failed to create NuGet package
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS: PacMAPSharp Cross-Platform NuGet Package Built!
echo ============================================
echo Package location: nupkg\
dir nupkg\*.nupkg
echo.
echo Binary structure (project base):
echo Windows binaries:
if exist "pacmap_enhanced.dll" (
    echo   ✓ pacmap_enhanced.dll
) else (
    echo   ✗ pacmap_enhanced.dll MISSING
)
if exist "libopenblas.dll" (
    echo   ✓ libopenblas.dll
) else (
    echo   ✗ libopenblas.dll MISSING
)
echo.
echo Linux binaries:
if exist "libpacmap_enhanced.so" (
    echo   ✓ libpacmap_enhanced.so (with proper linking)
) else (
    echo   ⚠ libpacmap_enhanced.so not available (Docker required for Linux build)
)
echo.

pause