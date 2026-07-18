@echo off
setlocal enabledelayedexpansion

set "SRC_DIR=%~dp0"
set "BUILD_DIR=D:\.my_llama"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

echo ===================================================
echo   Advanced CUDA Build Pipeline for RTX 5090 Blackwell
echo ===================================================
echo Source: !SRC_DIR!
echo Build:  !BUILD_DIR!
echo.

if "%~1"=="clean" (
    echo [INFO] Cleaning build directory...
    cd /d "%SystemRoot%"
    rmdir /s /q "!BUILD_DIR!"
    mkdir "!BUILD_DIR!"
    echo [SUCCESS] Build directory wiped clean and recreated.
    exit /b 0
)

if "%~1"=="cutlass" (
    echo [INFO] Step 1: Fetching and configuring CUTLASS headers...
    if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
    cd /d "!BUILD_DIR!"
    cmake -G "Visual Studio 18 2026" -A x64 -T "cuda=!CUDA_PATH!" -DFETCH_CUTLASS=ON -DFETCH_FLASHINFER=OFF -DISOLATED_BUILD=ON !SRC_DIR!
    if %errorlevel% neq 0 exit /b %errorlevel%
    echo [SUCCESS] CUTLASS successfully fetched and validated in cache!
    exit /b 0
)

if "%~1"=="flashinfer" (
    echo [INFO] Step 1: Fetching and configuring FlashInfer source...
    if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
    cd /d "!BUILD_DIR!"
    cmake -G "Visual Studio 18 2026" -A x64 -T "cuda=!CUDA_PATH!" -DFETCH_CUTLASS=OFF -DFETCH_FLASHINFER=ON -DISOLATED_BUILD=ON !SRC_DIR!
    if %errorlevel% neq 0 exit /b %errorlevel%

    echo [INFO] Step 2: Compiling FlashInfer heavy objects to static lib...
    cmake --build . --target cuda_kernels --config Release --parallel 16
    if %errorlevel% neq 0 exit /b %errorlevel%

    echo [SUCCESS] FlashInfer successfully downloaded and compiled into static lib!
    exit /b 0
)

if not exist "!CUDA_PATH!" (
    echo [ERROR] CUDA Toolkit v13.3 not found at !CUDA_PATH!
    exit /b 1
)

if "%~1"=="deps" (
    echo [INFO] Performing TOTAL deep wipe of build directory...
    cd /d "%SystemRoot%"
    rmdir /s /q "!BUILD_DIR!"
    mkdir "!BUILD_DIR!"
    goto :init_dir
)

:init_dir
if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"

if exist CMakeCache.txt (
    echo [INFO] Rapid compilation mode - All kernels
    goto :compile_stage
)

echo [INFO] Configuring full project - All dependencies enabled
cmake -G "Visual Studio 18 2026" -A x64 -T "cuda=!CUDA_PATH!" -DFETCH_CUTLASS=ON -DFETCH_FLASHINFER=ON -DISOLATED_BUILD=OFF !SRC_DIR!
if %errorlevel% neq 0 exit /b %errorlevel%

:compile_stage
echo [INFO] Compiling final target - Parallel Workers 16
cmake --build . --target cuda_kernels --config Release --parallel 16
if %errorlevel% neq 0 exit /b %errorlevel%

echo [SUCCESS] Full build completed. Static library is ready.
pause
