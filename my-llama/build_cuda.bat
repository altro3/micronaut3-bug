@echo off
setlocal enabledelayedexpansion

:: Определение путей
set "RAW_SRC=%~dp0"
if "!RAW_SRC:~-1!"=="\" set "RAW_SRC=!RAW_SRC:~0,-1!"

set "SRC_DIR=!RAW_SRC!"
set "BUILD_DIR=D:\.b_llama"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

echo ===================================================
echo   CUDA Build Pipeline for RTX 5090 (sm_120)
echo ===================================================
echo Source: !SRC_DIR!
echo Build:  !BUILD_DIR!
echo.

if not exist "!CUDA_PATH!" (
    echo [ERROR] CUDA Toolkit v13.3 not found at !CUDA_PATH!
    exit /b 1
)

:: Если передан аргумент "clean", то удаляем папку целиком (включая скачанный CUTLASS)
if "%1"=="clean" (
    echo [INFO] Performing TOTAL clean wipe as requested...
    if exist "!BUILD_DIR!" rmdir /s /q "!BUILD_DIR!"
)

if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"

:: мы удаляем ТОЛЬКО кэш переменных CMakeCache.txt и файлы сборки твоих собственных ядер.
echo [INFO] Refreshing CMake cache...
if exist CMakeCache.txt del /f /q CMakeCache.txt

:: Удаляем старые объектные файлы твоих ядер, чтобы гарантировать их пересборку,
:: но не трогаем служебную подпапку _deps, где лежит скачанный CUTLASS.
if exist backends\cuda\cuda_kernels.dir rmdir /s /q backends\cuda\cuda_kernels.dir
if exist backends\cuda\Release rmdir /s /q backends\cuda\Release

echo [INFO] Running CMake configuration...
cmake -T "cuda=!CUDA_PATH!" -DBACKEND=CUDA "!SRC_DIR!"

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed!
    exit /b %errorlevel%
)

echo [INFO] Compiling static library (Fast Incremental Mode)...
cmake --build . --config Release --parallel 16

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    exit /b %errorlevel%
)

echo [SUCCESS] CUDA kernels compiled perfectly! Static library is ready.
pause
