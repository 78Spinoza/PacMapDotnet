@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Publishing PacMAPSharp to NuGet
echo ============================================

rem Check if API key is provided
if "%1"=="" (
    echo ERROR: NuGet API key required
    echo Usage: publish_nuget.bat [API_KEY]
    echo.
    echo Get your API key from: https://www.nuget.org/account/apikeys
    pause
    exit /b 1
)

set "API_KEY=%1"

rem Build package first
echo [1/3] Building package...
call build_nuget.bat
if errorlevel 1 (
    echo ERROR: Failed to build package
    exit /b 1
)

rem Find the latest package
echo [2/3] Finding package to publish...
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

rem Publish to NuGet
echo [3/3] Publishing to NuGet.org...
dotnet nuget push "nupkg\%PACKAGE%" --api-key %API_KEY% --source https://api.nuget.org/v3/index.json
if errorlevel 1 (
    echo ERROR: Failed to publish package
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS: PacMAPSharp published to NuGet!
echo ============================================
echo Package: %PACKAGE%
echo URL: https://www.nuget.org/packages/PacMAPSharp/

pause