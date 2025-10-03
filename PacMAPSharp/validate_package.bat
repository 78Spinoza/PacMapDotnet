@echo off
echo ============================================
echo Validating PacMAPSharp Package
echo ============================================

rem Build package
echo [1/4] Building package...
call "./build_nuget.bat"
if errorlevel 1 (
    echo ERROR: Failed to build package
    exit /b 1
)

rem Find the latest package
echo [2/4] Finding package to validate...
for /f %%i in ('dir /b /o-d nupkg\PacMAPSharp.*.nupkg') do (
    set "PACKAGE=%%i"
    goto :found
)
:found

if not defined PACKAGE (
    echo ERROR: No package found in nupkg\
    exit /b 1
)

echo Found package: %PACKAGE%

rem Extract package contents
echo [3/4] Extracting package contents...
if exist temp_extract rmdir /s /q temp_extract
mkdir temp_extract
cd temp_extract
tar -xf "..\nupkg\%PACKAGE%"
if errorlevel 1 (
    echo ERROR: Failed to extract package
    cd ..
    exit /b 1
)

echo.
echo Package Contents:
dir /s
echo.

rem Validate required files
echo [4/4] Validating package structure...
set VALID=1

if not exist "lib\net8.0\PacMAPSharp.dll" (
    echo ERROR: Missing managed library: lib\net8.0\PacMAPSharp.dll
    set VALID=0
)

if not exist "pacmap_enhanced.dll" (
    echo ERROR: Missing native library: pacmap_enhanced.dll
    set VALID=0
)

if not exist "libopenblas.dll" (
    echo ERROR: Missing OpenBLAS library: libopenblas.dll
    set VALID=0
)

if not exist "libpacmap_enhanced.so" (
    echo WARNING: Missing Linux native library: libpacmap_enhanced.so
)

if not exist "libopenblas.so.0" (
    echo WARNING: Missing Linux OpenBLAS library: libopenblas.so.0
)

if not exist "PacMAPSharp.nuspec" (
    echo ERROR: Missing package manifest: PacMAPSharp.nuspec
    set VALID=0
)

cd ..
rmdir /s /q temp_extract

if %VALID%==1 (
    echo.
    echo ============================================
    echo SUCCESS: Package validation passed!
    echo ============================================
    echo Package: %PACKAGE%
    echo All required files present and accounted for.
) else (
    echo.
    echo ============================================
    echo ERROR: Package validation failed!
    echo ============================================
    echo Package: %PACKAGE%
    echo Please fix the missing files and rebuild.
    exit /b 1
)

pause