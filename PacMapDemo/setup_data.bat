@echo off
echo Setting up data for PacMAP Demo...

REM Create Data directory if it doesn't exist
if not exist "Data" mkdir Data

REM Copy data files from parent directory
echo Copying mammoth_data.csv...
copy "..\Data\mammoth_data.csv" "Data\" >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy mammoth_data.csv
    exit /b 1
)

echo Copying MNIST files...
copy "..\Data\mnist_images.npy" "Data\" >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy mnist_images.npy
    exit /b 1
)

copy "..\Data\mnist_labels.npy" "Data\" >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to copy mnist_labels.npy
    exit /b 1
)

echo Data setup completed successfully!
echo.
echo Files available:
dir Data /b

echo.
echo You can now run: dotnet run