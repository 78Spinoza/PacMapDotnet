@echo off
REM Set console to handle UTF-8 properly (suppress output)
chcp 65001 >nul 2>&1

echo ===========================================
echo   Building PACMAP C++ Library
echo ===========================================
echo.

REM Check if build directory exists, clean if needed
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

echo Creating build directory...
mkdir build
cd build

echo.
echo Configuring PACMAP with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo Building Release version...
cmake --build . --config Release
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   Build completed successfully!
echo ===========================================
echo.
echo Generated files:
echo   - pacmap.dll (shared library for C#)
echo   - pacmap.lib (static library)
echo.

echo.
echo ===========================================
echo   Copying DLL to C# Project
echo ===========================================
echo.

REM Copy DLL to C# project
if exist "Release\pacmap.dll" (
    if exist "..\PACMAPCSharp\PACMAPCSharp\" (
        copy "Release\pacmap.dll" "..\PACMAPCSharp\PACMAPCSharp\" >nul
        echo [COPY] pacmap.dll copied to C# project folder
    ) else (
        echo [WARN] C# project folder not found: ..\PACMAPCSharp\PACMAPCSharp\
    )
) else (
    echo [FAIL] pacmap.dll not found for copying
)

echo.
echo ===========================================
echo   PACMAP Build Summary
echo ===========================================
echo.
echo Your PACMAP C++ library is ready for C# integration!
echo.
echo Key files for C# P/Invoke:
echo   [FILE] build\Release\pacmap.dll      - PACMAP shared library
echo   [FILE] pacmap_simple_wrapper.h      - PACMAP header with all functions
echo.
echo PACMAP features available:
echo   [YES] Model Training              - pacmap_fit() with enhanced parameters
echo   [YES] Model Persistence           - pacmap_save_model() / pacmap_load_model()
echo   [YES] Data Transformation         - pacmap_transform() for out-of-sample projection
echo   [YES] Model Information           - pacmap_get_model_info() with metrics
echo   [YES] Cross-platform              - Windows ready, Linux compatible
echo.
echo Ready for C# integration!
echo Build process completed!
echo Press any key to exit...
pause >nul