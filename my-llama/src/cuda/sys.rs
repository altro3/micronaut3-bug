#![allow(non_snake_case)]
use std::ffi::c_void;

unsafe extern "C" {
    // VRAM на GPU
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub fn cudaMemsetAsync(dev_ptr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;

    // DMA копирование по PCIe
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;

    // Pinned memory на CPU
    pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    pub fn cudaFreeHost(ptr: *mut c_void) -> i32;

    // Streams
    pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

    // Events
    pub fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    pub fn cudaEventDestroy(event: *mut c_void) -> i32;
    pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    pub fn cudaEventQuery(event: *mut c_void) -> i32;
    pub fn cudaEventSynchronize(event: *mut c_void) -> i32;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
}

pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

pub const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;
pub const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;
pub const CUDA_HOST_ALLOC_DEFAULT: u32 = 0x00;
