@echo off
echo Building minimal PACMAP library...

REM Create build directory
if not exist build_minimal mkdir build_minimal
cd build_minimal

REM Configure with minimal settings
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DBUILD_TESTS=OFF ^
    -DBUILD_EXAMPLES=OFF

REM Build the minimal library
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    cd ..
    exit /b 1
)

REM Copy DLL to C# project
if exist "Release\pacmap.dll" (
    if exist "..\PACMAPCSharp\PACMAPCSharp\" (
        copy "Release\pacmap.dll" "..\PACMAPCSharp\PACMAPCSharp\" >nul
        echo [COPY] pacmap.dll copied to C# project folder
    ) else (
        echo [WARN] C# project folder not found
    )
)

echo Build completed successfully!
echo Location: build_minimal\Release\pacmap.dll
cd ..