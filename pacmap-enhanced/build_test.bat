@echo off
echo ===========================================
echo   Building and Testing All 6 Test Suites
echo   PacMAP Enhanced Comprehensive Validation
echo ===========================================
echo.

REM Set environment for Windows Rust compilation
set RUST_BACKTRACE=1

echo Checking Rust installation...
rustc --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Rust not available! Please install Rust from https://rustup.rs/
    pause
    exit /b 1
)

echo Rust compiler found, proceeding with build...
echo.

echo Building PacMAP Enhanced in Debug mode (avoiding DLL issues)...
cargo build
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ===========================================
echo   Running All 6 Comprehensive Tests
echo ===========================================
echo.

set ALL_TESTS_PASSED=1

echo [TEST 1] Running standard comprehensive validation...
cargo test test_standard_comprehensive -- --nocapture
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Test 1: Standard comprehensive test PASSED
) else (
    echo [FAIL] Test 1: Standard comprehensive test FAILED with code %ERRORLEVEL%
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 2] Running enhanced wrapper validation...
cargo test test_enhanced_wrapper -- --nocapture
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Test 2: Enhanced wrapper test PASSED
) else (
    echo [FAIL] Test 2: Enhanced wrapper test FAILED with code %ERRORLEVEL%
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 3] Running comprehensive pipeline validation...
cargo test test_comprehensive_pipeline -- --nocapture
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Test 3: Comprehensive pipeline test PASSED
) else (
    echo [FAIL] Test 3: Comprehensive pipeline test FAILED with code %ERRORLEVEL%
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 4] Running error fixes validation...
cargo test test_error_fixes_simple -- --nocapture
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Test 4: Error fixes test PASSED
) else (
    echo [FAIL] Test 4: Error fixes test FAILED with code %ERRORLEVEL%
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 5] Running metric validation...
cargo test test_metric_validation -- --nocapture
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Test 5: Metric validation test PASSED
) else (
    echo [FAIL] Test 5: Metric validation test FAILED with code %ERRORLEVEL%
    set ALL_TESTS_PASSED=0
)

echo.
echo [TEST 6] Running quantization comprehensive validation...
cargo test test_quantization_comprehensive -- --nocapture
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Test 6: Quantization comprehensive test PASSED
) else (
    echo [FAIL] Test 6: Quantization comprehensive test FAILED with code %ERRORLEVEL%
    set ALL_TESTS_PASSED=0
)

echo.
echo ===========================================
echo   Test Results Summary
echo ===========================================
echo.
if %ALL_TESTS_PASSED% EQU 1 (
    echo [SUCCESS] ALL 6 TESTS PASSED!
    echo ✅ Standard comprehensive validation
    echo ✅ Enhanced wrapper validation
    echo ✅ Comprehensive pipeline validation
    echo ✅ Error fixes validation
    echo ✅ Metric validation
    echo ✅ Quantization comprehensive validation
    echo.
    echo PacMAP Enhanced comprehensive test suite fully validated!
    echo.
    echo Key features verified:
    echo   - HNSW neighbor search integration
    echo   - Quantization with parameter preservation
    echo   - Comprehensive normalization modes
    echo   - FFI wrapper functionality
    echo   - Model persistence and save/load
    echo   - Cross-platform compatibility
) else (
    echo [FAILURE] SOME TESTS FAILED!
    echo ❌ Check test output above for details
    echo.
    echo Build may have issues - review failed tests!
)

echo.
echo Press any key to exit...
pause >nul