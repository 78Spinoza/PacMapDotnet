@echo off
echo ============================================
echo Verifying PacMAPSharp Native Binaries
echo ============================================

rem Check if Rust DLL exists and get version
echo [1/3] Checking Rust library...
if exist "pacmap_enhanced.dll" (
    echo ✓ pacmap_enhanced.dll found
    echo   Size:
    dir /b pacmap_enhanced.dll | xargs ls -l
) else (
    echo ✗ pacmap_enhanced.dll NOT FOUND
    echo   Building from source...
    cd ..\pacmap-enhanced
    cargo build --release
    if errorlevel 1 (
        echo ERROR: Failed to build Rust library
        cd ..\PacMAPSharp
        exit /b 1
    )
    copy "target\release\pacmap_enhanced.dll" "..\PacMAPSharp\pacmap_enhanced.dll" /Y
    cd ..\PacMAPSharp
    echo ✓ pacmap_enhanced.dll built and copied
)

rem Check OpenBLAS dependency
echo [2/3] Checking OpenBLAS dependency...
if exist "libopenblas.dll" (
    echo ✓ libopenblas.dll found
    echo   Size:
    dir /b libopenblas.dll | xargs ls -l
) else if exist "..\lapack-binaries\bin\libopenblas.dll" (
    echo ✓ libopenblas.dll found in lapack-binaries
    copy "..\lapack-binaries\bin\libopenblas.dll" "libopenblas.dll" /Y
    echo ✓ libopenblas.dll copied
) else (
    echo ✗ libopenblas.dll NOT FOUND
    echo   Please ensure OpenBLAS binaries are available
    exit /b 1
)

rem Test library loading (basic)
echo [3/3] Testing library compatibility...
echo Creating test program...

echo using System; > test_load.cs
echo using System.Runtime.InteropServices; >> test_load.cs
echo. >> test_load.cs
echo class Test { >> test_load.cs
echo     [DllImport("pacmap_enhanced.dll", CallingConvention = CallingConvention.Cdecl)] >> test_load.cs
echo     private static extern IntPtr pacmap_get_version(); >> test_load.cs
echo. >> test_load.cs
echo     static void Main() { >> test_load.cs
echo         try { >> test_load.cs
echo             var versionPtr = pacmap_get_version(); >> test_load.cs
echo             if (versionPtr != IntPtr.Zero) { >> test_load.cs
echo                 var version = Marshal.PtrToStringUTF8(versionPtr); >> test_load.cs
echo                 Console.WriteLine("✓ Library loaded successfully"); >> test_load.cs
echo                 Console.WriteLine("  Version: " + version); >> test_load.cs
echo             } else { >> test_load.cs
echo                 Console.WriteLine("✗ Version function returned null"); >> test_load.cs
echo             } >> test_load.cs
echo         } catch (Exception e) { >> test_load.cs
echo             Console.WriteLine("✗ Failed to load library: " + e.Message); >> test_load.cs
echo         } >> test_load.cs
echo     } >> test_load.cs
echo } >> test_load.cs

csc /platform:x64 test_load.cs
if errorlevel 1 (
    echo ERROR: Failed to compile test program
    goto cleanup
)

echo Running library test...
test_load.exe
set TEST_RESULT=%errorlevel%

:cleanup
if exist test_load.cs del test_load.cs
if exist test_load.exe del test_load.exe

if %TEST_RESULT%==0 (
    echo.
    echo ============================================
    echo SUCCESS: All binaries verified!
    echo ============================================
) else (
    echo.
    echo ============================================
    echo ERROR: Binary verification failed!
    echo ============================================
    exit /b 1
)

pause