use libc::c_int;
use std::ffi::c_void;
use std::ptr::null_mut;

unsafe extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    pub fn cudaMemsetAsync(
        dev_ptr: *mut c_void,
        value: c_int,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;

    pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    pub fn cudaFreeHost(ptr: *mut c_void) -> c_int;

    pub fn cudaStreamCreate(stream: *mut *mut c_void) -> c_int;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
}

pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_HOST_ALLOC_DEFAULT: u32 = 0x00;

/// RAII-обертка над асинхронным CUDA Stream для выполнения ядер без блокировки CPU
pub struct CudaStream {
    raw: *mut c_void,
}

impl CudaStream {
    pub fn new() -> Self {
        let mut raw = null_mut();
        unsafe {
            let status = cudaStreamCreate(&mut raw);
            if status != 0 {
                panic!("Критическая ошибка CUDA: Не удалось создать асинхронный Stream");
            }
        }
        CudaStream { raw }
    }

    pub fn synchronize(&self) {
        unsafe {
            let status = cudaStreamSynchronize(self.raw);
            if status != 0 {
                panic!("Ошибка синхронизации CUDA Stream");
            }
        }
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                cudaStreamDestroy(self.raw);
            }
        }
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}
