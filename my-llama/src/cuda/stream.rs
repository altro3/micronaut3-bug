use crate::cuda::sys::{CUDA_STREAM_NON_BLOCKING, cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamSynchronize};
use std::ffi::c_void;
use std::ptr::null_mut;

pub struct CudaStream {
    raw: *mut c_void,
}

impl CudaStream {
    pub fn new() -> Self {
        let mut raw = null_mut();
        unsafe {
            if cudaStreamCreateWithFlags(&mut raw, CUDA_STREAM_NON_BLOCKING) != 0 {
                panic!("Критическая ошибка CUDA: Не удалось создать асинхронный Stream");
            }
        }
        CudaStream { raw }
    }

    pub fn synchronize(&self) {
        unsafe {
            if cudaStreamSynchronize(self.raw) != 0 {
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
                let _ = cudaStreamDestroy(self.raw);
            }
        }
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}
