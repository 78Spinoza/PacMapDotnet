@echo off
echo ===========================================
echo   Building PacMAP Builder Docker Image
echo ===========================================
echo.
echo This will create a reusable Docker image with all dependencies.
echo You only need to run this ONCE.
echo.

REM Check Docker
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker not available
    pause
    exit /b 1
)

echo [1/2] Building pacmap-builder image (this may take 5-10 minutes)...
echo This includes: Ubuntu, Rust, OpenBLAS, build tools
echo.

docker build -f Dockerfile.pacmap-builder -t pacmap-builder:latest .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build pacmap-builder image
    pause
    exit /b 1
)

echo.
echo [2/2] Testing the image...
docker run --rm pacmap-builder:latest rustc --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Image test failed
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   SUCCESS: PacMAP Builder Image Ready!
echo ===========================================
echo.
echo Image name: pacmap-builder:latest
echo You can now use fast Linux builds!
echo.
echo To remove the image later:
echo   docker rmi pacmap-builder:latest
echo.
pause