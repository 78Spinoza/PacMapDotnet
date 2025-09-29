@echo off
echo ============================================
echo Building PacMAPSharp NuGet Package
echo ============================================

rem Clean previous builds
echo [1/5] Cleaning previous builds...
if exist bin rmdir /s /q bin
if exist obj rmdir /s /q obj
if exist nupkg rmdir /s /q nupkg

rem Build Rust core library first
echo [2/5] Building Rust core library...
cd ..\pacmap-enhanced
cargo build --release
if errorlevel 1 (
    echo ERROR: Failed to build Rust library
    cd ..\PacMAPSharp
    exit /b 1
)
cd ..\PacMAPSharp

rem Copy latest native binaries
echo [3/5] Copying native binaries...
copy "..\pacmap-enhanced\target\release\pacmap_enhanced.dll" "pacmap_enhanced.dll" /Y
if not exist "pacmap_enhanced.dll" (
    echo ERROR: Failed to copy Rust DLL
    exit /b 1
)

copy "..\lapack-binaries\bin\libopenblas.dll" "libopenblas.dll" /Y
if not exist "libopenblas.dll" (
    echo ERROR: Failed to copy OpenBLAS DLL
    exit /b 1
)

rem Build C# library
echo [4/5] Building C# library...
dotnet build --configuration Release
if errorlevel 1 (
    echo ERROR: Failed to build C# library
    exit /b 1
)

rem Pack NuGet
echo [5/5] Creating NuGet package...
dotnet pack --configuration Release --no-build --output nupkg
if errorlevel 1 (
    echo ERROR: Failed to create NuGet package
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS: PacMAPSharp NuGet package built!
echo ============================================
echo Package location: nupkg\
dir nupkg\*.nupkg

pause