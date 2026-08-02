#pragma once
#include "core_api.h"

extern "C" KERNEL_API void launch_cute_blackwell_gemm(float *d_C, const void *d_A, const void *d_B);
