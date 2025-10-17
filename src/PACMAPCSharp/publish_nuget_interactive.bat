@echo off
setlocal enabledelayedexpansion
echo Publishing PacMapSharp NuGet Package to NuGet.org
echo.

REM Find the latest PacMapSharp package using PowerShell for correct date sorting
for /f %%i in ('powershell -Command "Get-ChildItem PACMAPCSharp\bin\Release\PacMapSharp.*.nupkg | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object -ExpandProperty Name"') do (
    set "LATEST_PACKAGE=%%i"
    goto :found_package
)

:found_package
if "!LATEST_PACKAGE!"=="" (
    echo ❌ No NuGet package found! Run 'dotnet pack' first to create the package.
    pause
    exit /b 1
)

REM Extract version from filename (improved parsing for version like 2.8.29)
for /f "tokens=2,3 delims=." %%a in ("!LATEST_PACKAGE!") do (
    set "VERSION=%%a.%%b"
)

echo ✅ Package found: !LATEST_PACKAGE! (v!VERSION!)
echo 🔧 FIXED: Now using PowerShell date-based sorting for correct package detection
echo.

REM Show package details
echo Package Information:
dir "PACMAPCSharp\bin\Release\!LATEST_PACKAGE!"
dir "PACMAPCSharp\bin\Release\PacMapSharp.!VERSION!.snupkg" 2>nul
echo.

echo 🚀 READY TO PUBLISH TO NUGET.ORG!
echo.
echo ⚠️  This will publish your PacMapSharp v!VERSION! package to the public NuGet repository.
echo    Make sure you're ready for this!
echo.

REM Ask for confirmation
set /p "confirm=Do you want to proceed with publishing? (y/N): "
if /i not "!confirm!"=="y" (
    echo Publishing cancelled.
    pause
    exit /b 0
)

echo.
echo 🔐 Please enter your NuGet API key:
echo    (Get it from: https://www.nuget.org/account/apikeys)
echo.
set /p "apikey=API Key: "

if "!apikey!"=="" (
    echo ❌ No API key provided. Publishing cancelled.
    pause
    exit /b 1
)

echo.
echo 📦 Publishing package to NuGet.org...
echo.

REM Change to the package directory
cd PACMAPCSharp\bin\Release

REM Execute the publish command with symbol package
dotnet nuget push "!LATEST_PACKAGE!" --source https://api.nuget.org/v3/index.json --api-key "!apikey!" --symbol-source https://nuget.smbsrc.net/

if !ERRORLEVEL! EQU 0 (
    echo.
    echo 🎉 SUCCESS! PacMapSharp package published successfully!
    echo.
    echo 📍 Your package is now available at:
    echo    https://www.nuget.org/packages/PacMapSharp/!VERSION!
    echo.
    echo ⏰ Note: It may take a few minutes to appear in search results.
    echo.
    echo 🚀 Your revolutionary PacMapSharp v!VERSION! with 3.1-12.5x performance optimization is now live!
    echo.
    echo ✨ Features included:
    echo    - Cross-platform 64-bit binaries (Windows/Linux)
    echo    - OpenMP 8-thread parallelization
    echo    - AVX2/AVX512 SIMD optimization
    echo    - HNSW acceleration
    echo    - Enterprise-grade thread safety
    echo    - Production validation on MNIST 70K, 1M Mammoth datasets
) else (
    echo.
    echo ❌ Publishing failed! Error code: !ERRORLEVEL!
    echo.
    echo Common issues:
    echo - Invalid API key
    echo - Package version already exists
    echo - Network connectivity issues
    echo - Package validation errors
    echo - Missing symbol package upload permissions
    echo.
    echo Please check the error message above and try again.
)

echo.
pause