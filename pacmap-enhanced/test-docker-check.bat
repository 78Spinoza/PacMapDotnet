@echo off
setlocal enabledelayedexpansion

echo Testing Docker check...
docker images pacmap-builder:latest --quiet >nul 2>nul
echo Error level: !ERRORLEVEL!
if !ERRORLEVEL! NEQ 0 (
    echo Image not found
) else (
    echo Image found
)

pause