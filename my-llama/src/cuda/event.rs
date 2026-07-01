use crate::cuda::stream::CudaStream;
use crate::cuda::sys::{
    cudaEventCreateWithFlags, cudaEventDestroy, cudaEventElapsedTime, cudaEventQuery, cudaEventRecord, cudaEventSynchronize,
    CUDA_EVENT_DISABLE_TIMING,
};
use std::ffi::c_void;
use std::ptr::null_mut;

pub struct CudaEvent {
    raw: *mut c_void,
}

impl CudaEvent {
    pub fn new(disable_timing: bool) -> Self {
        let mut raw = null_mut();
        let flags = if disable_timing { CUDA_EVENT_DISABLE_TIMING } else { 0 };
        unsafe {
            if cudaEventCreateWithFlags(&mut raw, flags) != 0 {
                panic!("Ошибка CUDA: Не удалось инициализировать Event");
            }
        }
        CudaEvent { raw }
    }

    pub fn record(&self, stream: &CudaStream) {
        unsafe {
            let _ = cudaEventRecord(self.raw, stream.as_raw());
        }
    }

    pub fn is_ready(&self) -> bool {
        unsafe { cudaEventQuery(self.raw) == 0 }
    }

    pub fn synchronize(&self) {
        unsafe {
            if cudaEventSynchronize(self.raw) != 0 {
                panic!("Ошибка жесткой синхронизации по CUDA Event");
            }
        }
    }

    pub fn elapsed_time_ms(start: &Self, end: &Self) -> f32 {
        let mut ms = 0.0;
        unsafe {
            let _ = cudaEventElapsedTime(&mut ms, start.raw, end.raw);
        }
        ms
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = cudaEventDestroy(self.raw);
            }
        }
    }
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}
