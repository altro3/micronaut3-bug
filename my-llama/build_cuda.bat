@echo off
setlocal enabledelayedexpansion

set "RAW_SRC=%~dp0"
if "!RAW_SRC:~-1!"=="\" set "RAW_SRC=!RAW_SRC:~0,-1!"

set "SRC_DIR=!RAW_SRC!"
set "BUILD_DIR=D:\.b_llama"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

echo ===================================================
echo   CUDA Build Pipeline for RTX 5090 (sm_100)
echo ===================================================
echo Source: !SRC_DIR!
echo Build:  !BUILD_DIR!
echo.

if not exist "!CUDA_PATH!" (
    echo [ERROR] CUDA Toolkit v13.3 not found at !CUDA_PATH!
    exit /b 1
)

if exist "!BUILD_DIR!" (
    echo [INFO] Wiping old CMake cache...
    cd /d "!BUILD_DIR!"
    if exist CMakeCache.txt del /f /q CMakeCache.txt
    if exist CMakeFiles rmdir /s /q CMakeFiles
) else (
    echo [INFO] Creating build directory...
    mkdir "!BUILD_DIR!"
)

cd /d "!BUILD_DIR!"

echo [INFO] Running CMake configuration...
cmake -T "cuda=!CUDA_PATH!" -DBACKEND=CUDA "!SRC_DIR!"

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed!
    exit /b %errorlevel%
)

echo [INFO] Compiling static library on 16 cores...
cmake --build . --config Release --parallel 16

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    exit /b %errorlevel%
)

echo [SUCCESS] CUDA kernels compiled perfectly! Static library is ready.
pause
