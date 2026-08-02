#pragma once

#ifdef _WIN32
#ifdef EXPORT_KERNELS
#define KERNEL_API __declspec(dllexport)
#else
#define KERNEL_API __declspec(dllimport)
#endif
#else
#define KERNEL_API __attribute__((visibility("default")))
#endif
