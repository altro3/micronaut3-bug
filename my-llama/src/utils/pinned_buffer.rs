use crate::utils::cuda_stream::{cudaFreeHost, cudaHostAlloc, CUDA_HOST_ALLOC_DEFAULT};
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
                cudaFreeHost(self.raw_ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_host_buffer_dma() {
        let elements = 5;
        let mut pinned_buf = PinnedHostBuffer::new(elements);

        {
            let slice = pinned_buf.as_slice_mut();
            assert_eq!(slice.len(), elements);
            slice[0] = 7.7f32;
            slice[4] = 9.9f32;
        }

        assert!(!pinned_buf.as_ptr().is_null());
    }
}
