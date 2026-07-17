@echo off
setlocal enabledelayedexpansion

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

:: Режим тотальной очистки (вызывается вручную как build_cuda.bat clean)
if "%1"=="clean" (
    echo [INFO] Performing TOTAL clean wipe as requested...
    if exist "!BUILD_DIR!" rmdir /s /q "!BUILD_DIR!"
)

if not exist "!BUILD_DIR!" mkdir "!BUILD_DIR!"
cd /d "!BUILD_DIR!"

:: Удаляем ТОЛЬКО файл кэша CMakeCache.txt, чтобы CMake видел
:: добавление новых файлов в структуру, но НЕ трогаем папки с .obj файлами!
echo [INFO] Refreshing CMake configuration...
if exist CMakeCache.txt del /f /q CMakeCache.txt

echo [INFO] Running CMake configuration...
cmake -T "cuda=!CUDA_PATH!" -DBACKEND=CUDA "!SRC_DIR!"

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed!
    exit /b %errorlevel%
)

:: Нативная проверка изменений от MSBuild. Сборка пойдет на 16 ядрах.
echo [INFO] Compiling static library (Native Incremental Mode)...
cmake --build . --config Release --parallel 16

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    exit /b %errorlevel%
)

echo [SUCCESS] CUDA kernels compiled perfectly! Static library is ready.
pause
