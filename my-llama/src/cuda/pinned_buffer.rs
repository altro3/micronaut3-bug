use crate::cuda::sys::{CUDA_HOST_ALLOC_DEFAULT, cudaFreeHost, cudaHostAlloc};
use std::ffi::c_void;

pub struct PinnedHostBuffer {
    raw_ptr: *mut c_void,
    elements: usize,
}

impl PinnedHostBuffer {
    pub fn new(elements: usize) -> Self {
        let size_in_bytes = elements * size_of::<f32>();
        let mut raw_ptr = std::ptr::null_mut();
        unsafe {
            if cudaHostAlloc(&mut raw_ptr, size_in_bytes, CUDA_HOST_ALLOC_DEFAULT) != 0 {
                panic!("Критическая ошибка: Выделение Pinned Memory на CPU провалено");
            }
        }
        PinnedHostBuffer { raw_ptr, elements }
    }

    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.raw_ptr as *mut f32, self.elements) }
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.raw_ptr
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.raw_ptr.is_null() {
            unsafe {
                let _ = cudaFreeHost(self.raw_ptr);
            }
        }
    }
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}
