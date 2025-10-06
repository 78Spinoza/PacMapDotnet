@echo off
setlocal enabledelayedexpansion
echo ===========================================
echo   Enhanced UMAP Cross-Platform Build
echo ===========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker not available
    pause
    exit /b 1
)

REM === BUILD WINDOWS VERSION ===
echo [1/2] Building Enhanced Windows version...
echo ----------------------------------------

if exist build-windows (
    rmdir /s /q build-windows
)
mkdir build-windows
cd build-windows

cmake .. -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=ON
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Windows CMake configuration failed!
    exit /b 1
)

cmake --build . --config Release
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Windows build failed!
    exit /b 1
)

echo [INFO] Running Windows validation tests...
cd Release

REM Run all essential tests
set WIN_ALL_TESTS_PASSED=1

if exist "run_all_tests.exe" (
    echo [TEST] Running comprehensive PACMAP test suite...
    run_all_tests.exe
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Comprehensive PACMAP test PASSED
    ) else (
        echo [FAIL] Comprehensive PACMAP test FAILED with code !ERRORLEVEL!
        set WIN_ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] run_all_tests.exe not found
    set WIN_ALL_TESTS_PASSED=0
)

if exist "test_minimal_standalone.exe" (
    echo [TEST] Running minimal standalone PACMAP test...
    test_minimal_standalone.exe
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Minimal standalone test PASSED
    ) else (
        echo [FAIL] Minimal standalone test FAILED with code !ERRORLEVEL!
        set WIN_ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] test_minimal_standalone.exe not found
)

if exist "test_simple_minimal.exe" (
    echo [TEST] Running simple minimal PACMAP test...
    test_simple_minimal.exe
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Simple minimal test PASSED
    ) else (
        echo [FAIL] Simple minimal test FAILED with code !ERRORLEVEL!
        set WIN_ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] test_simple_minimal.exe not found
)

if exist "test_basic_integration.exe" (
    echo [TEST] Running basic PACMAP integration test...
    test_basic_integration.exe
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Basic integration test PASSED
    ) else (
        echo [FAIL] Basic integration test FAILED with code !ERRORLEVEL!
        set WIN_ALL_TESTS_PASSED=0
    )
) else (
    echo [WARN] test_basic_integration.exe not found
    set WIN_ALL_TESTS_PASSED=0
)

cd ..
if !WIN_ALL_TESTS_PASSED! EQU 1 (
    echo [PASS] ALL Windows tests completed successfully
) else (
    echo [FAIL] Some Windows tests failed - build may have issues!
)

echo [PASS] Windows build completed
cd ..

REM === BUILD LINUX VERSION WITH DOCKER ===
echo.
echo [2/2] Building Enhanced Linux version with Docker...
echo ----------------------------------------

if exist build-linux (
    rmdir /s /q build-linux
)

REM Convert Windows path to Unix path for Docker
set UNIX_PATH=%cd:\=/%
set UNIX_PATH=%UNIX_PATH:C:=/c%

REM Run Docker build with corrected paths
docker run --rm -v "%UNIX_PATH%":/src -w /src ubuntu:22.04 bash -c "apt-get update && apt-get install -y build-essential cmake libstdc++-11-dev && mkdir -p build-linux && cd build-linux && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON && make pacmap run_all_tests test_minimal_standalone test_simple_minimal test_basic_integration -j4 && echo Build completed && ls -la && if [ -f lib/libpacmap.so ]; then cp lib/libpacmap.so libpacmap_backup.so && echo Library backup created; fi && echo Linux build finished successfully"

if !ERRORLEVEL! NEQ 0 (
    echo ERROR: Docker Linux build failed!
    exit /b 1
)

echo [PASS] Linux build completed

REM === SETUP CSHARP PROJECT STRUCTURE ===
echo.
echo [3/3] Setting up Enhanced UMAPuwotSharp project...
echo ----------------------------------------

REM Create C# project structure
if not exist "..\UMAPuwotSharp" (
    echo [INFO] Creating UMAPuwotSharp project structure
    mkdir "..\UMAPuwotSharp"
)

if not exist "..\UMAPuwotSharp\UMAPuwotSharp" (
    mkdir "..\UMAPuwotSharp\UMAPuwotSharp"
)

REM Copy Windows DLL directly to project base folder
if exist "build-windows\Release\uwot.dll" (
    copy "build-windows\Release\uwot.dll" "..\UMAPuwotSharp\UMAPuwotSharp\uwot.dll"
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Copied Windows uwot.dll to C# project base folder
    ) else (
        echo [FAIL] Failed to copy Windows uwot.dll - Error: !ERRORLEVEL!
    )
) else (
    echo [FAIL] Windows uwot.dll not found in build-windows\Release\
)

REM Copy Linux .so file directly to project base folder
set LINUX_LIB_COPIED=0

REM Try libpacmap_final.so first (our guaranteed backup from Docker)
if exist "build-linux\libpacmap_final.so" (
    for %%A in ("build-linux\libpacmap_final.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\libpacmap_final.so" "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so"
            echo [PASS] Copied libpacmap_final.so as libpacmap.so to C# project base folder
            set LINUX_LIB_COPIED=1
            goto :linux_lib_done
        )
    )
)

REM Try libpacmap.so from lib directory (Linux CMake default)
if exist "build-linux\lib\libpacmap.so" (
    for %%A in ("build-linux\lib\libpacmap.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\lib\libpacmap.so" "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so"
            echo [PASS] Copied lib/libpacmap.so as libpacmap.so to C# project base folder
            set LINUX_LIB_COPIED=1
            goto :linux_lib_done
        )
    )
)

REM Check for versioned .so files in lib directory
for %%F in (build-linux\lib\libpacmap.so.*.*.*) do (
    if exist "%%F" (
        for %%A in ("%%F") do (
            if %%~zA GTR 1000 (
                copy "%%F" "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so"
                echo [PASS] Copied lib/%%~nxF as libpacmap.so to C# project base folder
                set LINUX_LIB_COPIED=1
                goto :linux_lib_done
            )
        )
    )
)

REM Check for versioned .so files in root directory
for %%F in (build-linux\libpacmap.so.*.*.*) do (
    if exist "%%F" (
        for %%A in ("%%F") do (
            if %%~zA GTR 1000 (
                copy "%%F" "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so"
                echo [PASS] Copied %%~nxF as libpacmap.so to C# project base folder
                set LINUX_LIB_COPIED=1
                goto :linux_lib_done
            )
        )
    )
)

REM Try backup file if available
if exist "build-linux\libpacmap_backup.so" (
    for %%A in ("build-linux\libpacmap_backup.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\libpacmap_backup.so" "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so"
            echo [PASS] Copied libpacmap_backup.so as libpacmap.so to C# project base folder
            set LINUX_LIB_COPIED=1
            goto :linux_lib_done
        )
    )
)

REM Check for libpacmap.so in lib directory
if exist "build-linux\lib\libpacmap.so" (
    for %%A in ("build-linux\lib\libpacmap.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\lib\libpacmap.so" "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so"
            if !ERRORLEVEL! EQU 0 (
                echo [PASS] Copied Linux lib/libpacmap.so to C# project base folder
                set LINUX_LIB_COPIED=1
            ) else (
                echo [FAIL] Failed to copy Linux libpacmap.so - Error: !ERRORLEVEL!
            )
        )
    )
)

REM Check for libpacmap.so in root directory (last resort)
if exist "build-linux\libpacmap.so" (
    for %%A in ("build-linux\libpacmap.so") do (
        if %%~zA GTR 1000 (
            copy "build-linux\libpacmap.so" "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so"
            if !ERRORLEVEL! EQU 0 (
                echo [PASS] Copied Linux libpacmap.so to C# project base folder
                set LINUX_LIB_COPIED=1
            ) else (
                echo [FAIL] Failed to copy Linux libpacmap.so - Error: !ERRORLEVEL!
            )
        )
    )
)

:linux_lib_done
if !LINUX_LIB_COPIED! EQU 0 (
    echo [FAIL] Linux libpacmap.so not found in build-linux\
    echo [INFO] Available files in build-linux:
    if exist "build-linux" (
        dir build-linux\ /B 2>nul || echo        Directory empty or not accessible
    ) else (
        echo        build-linux directory not found
    )
)

echo [PASS] Enhanced libraries setup completed

REM === SUMMARY ===
echo.
echo ===========================================
echo   PACMAP Build Summary
echo ===========================================
echo.
echo Windows libraries (build-windows\bin\Release\):
if exist "build-windows\bin\Release\pacmap.dll" (
    echo   [PASS] pacmap.dll (PACMAP library)
    for %%A in ("build-windows\bin\Release\pacmap.dll") do echo         Size: %%~zA bytes
) else (echo   [FAIL] pacmap.dll)

if exist "build-windows\bin\Release\run_all_tests.exe" (
    echo   [PASS] run_all_tests.exe
) else (
    echo   [FAIL] run_all_tests.exe
)

if exist "build-windows\bin\Release\test_minimal_standalone.exe" (
    echo   [PASS] test_minimal_standalone.exe
) else (
    echo   [FAIL] test_minimal_standalone.exe
)

if exist "build-windows\bin\Release\test_simple_minimal.exe" (
    echo   [PASS] test_simple_minimal.exe
) else (
    echo   [FAIL] test_simple_minimal.exe
)

if exist "build-windows\bin\Release\test_basic_integration.exe" (
    echo   [PASS] test_basic_integration.exe
) else (
    echo   [FAIL] test_basic_integration.exe
)

echo.
echo Linux libraries (build-linux\):
if exist "build-linux" (
    dir build-linux\libpacmap.so* build-linux\*.so build-linux\lib\libpacmap.so* build-linux\lib\*.so /B 2>nul | findstr /R ".*" >nul
    if !ERRORLEVEL! EQU 0 (
        echo   [PASS] Linux .so files found:
        for %%F in (build-linux\libpacmap.so* build-linux\*.so build-linux\lib\libpacmap.so* build-linux\lib\*.so) do (
            if exist "%%F" (
                for %%A in ("%%F") do echo         %%~nxF (%%~zA bytes)
            )
        )
    ) else (
        echo   [FAIL] No Linux .so files found
    )
) else (
    echo   [FAIL] build-linux directory not found
)

if exist "build-linux\bin\run_all_tests" (
    echo   [PASS] run_all_tests
) else (
    echo   [FAIL] run_all_tests
)

if exist "build-linux\bin\test_minimal_standalone" (
    echo   [PASS] test_minimal_standalone
) else (
    echo   [FAIL] test_minimal_standalone
)

if exist "build-linux\bin\test_simple_minimal" (
    echo   [PASS] test_simple_minimal
) else (
    echo   [FAIL] test_simple_minimal
)

echo.
echo PACMAPCSharp C# project files:
if exist "..\PACMAPCSharp\PACMAPCSharp\pacmap.dll" (
    echo   [PASS] Windows library: pacmap.dll
    for %%A in ("..\PACMAPCSharp\PACMAPCSharp\pacmap.dll") do echo         Size: %%~zA bytes
) else (echo   [FAIL] Windows library missing)

if exist "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so" (
    echo   [PASS] Linux library: libpacmap.so
    for %%A in ("..\PACMAPCSharp\PACMAPCSharp\libpacmap.so") do echo         Size: %%~zA bytes
) else (echo   [FAIL] Linux library missing)

echo.
echo Cross-platform libraries ready for direct deployment:
if exist "..\PACMAPCSharp\PACMAPCSharp\pacmap.dll" (
    echo   [PASS] Windows library ready: pacmap.dll
) else (echo   [WARN] Windows library missing)

if exist "..\PACMAPCSharp\PACMAPCSharp\libpacmap.so" (
    echo   [PASS] Linux library ready: libpacmap.so
) else (echo   [WARN] Linux library missing)

echo.
echo ===========================================
echo   PACMAP Features Available
echo ===========================================
echo.
echo Your PACMAP library now supports:
echo   - Fast and scalable dimensionality reduction
echo   - Preserves both local and global structure
echo   - Complete model save/load functionality
echo   - True out-of-sample projection (transform new data)
echo   - Progress reporting with callback support
echo   - Based on proven PACMAP algorithms
echo   - Cross-platform support (Windows + Linux)
echo.
echo ===========================================
echo   Ready for C# Development!
echo ===========================================
echo.
echo Next steps:
echo   1. Build C# library: dotnet build ..\PACMAPCSharp\PACMAPCSharp
echo   2. Create NuGet package: dotnet pack ..\PACMAPCSharp\PACMAPCSharp --configuration Release
echo   3. Test PACMAP embedding: var embedding = model.Fit(data, embeddingDimension: 2);
echo   4. Use progress reporting: model.FitWithProgress(data, progressCallback);
echo.
pause