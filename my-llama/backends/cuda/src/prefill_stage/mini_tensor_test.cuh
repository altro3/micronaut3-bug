#pragma once

#ifdef _WIN32
#ifdef EXPORT_KERNELS
#define KERNEL_API __declspec(dllexport)
#else
#define KERNEL_API __declspec(dllimport)
#endif
#else
#define KERNEL_API
#endif

extern "C" KERNEL_API void launch_cute_blackwell_gemm(float *d_C, const void *d_A, const void *d_B);
