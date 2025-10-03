@echo off
echo Testing HNSW Determinism - Two identical runs should produce identical results
echo ============================================================================
echo.
echo Run 1:
"C:\PacMAN\PacMapDemo\bin\Debug\net8.0\PacMapDemo.exe" mammoth hnsw > test_run1.txt 2>&1

echo.
echo Run 2:
"C:\PacMAN\PacMapDemo\bin\Debug\net8.0\PacMapDemo.exe" mammoth hnsw > test_run2.txt 2>&1

echo.
echo Comparing results:
fc test_run1.txt test_run2.txt

echo.
echo Done. Check for differences above.
pause