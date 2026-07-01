use crate::cuda::stream::CudaStream;
use crate::cuda::sys::{cudaFree, cudaMalloc, cudaMemcpyAsync, cudaMemsetAsync, CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE};
use std::ffi::c_void;

pub struct CudaBuffer {
    raw_ptr: *mut c_void,
    size_in_bytes: usize,
    elements: usize,
    is_owner: bool,
}

impl CudaBuffer {
    pub fn new(elements: usize) -> Self {
        let size_in_bytes = elements * size_of::<f32>();
        let mut raw_ptr = std::ptr::null_mut();
        unsafe {
            if cudaMalloc(&mut raw_ptr, size_in_bytes) != 0 {
                panic!("Недостаточно VRAM: Выделение {} байт (f32) провалено", size_in_bytes);
            }
        }
        CudaBuffer {
            raw_ptr,
            size_in_bytes,
            elements,
            is_owner: true,
        }
    }

    pub fn new_int(elements: usize) -> Self {
        let size_in_bytes = elements * size_of::<i32>();
        let mut raw_ptr = std::ptr::null_mut();
        unsafe {
            if cudaMalloc(&mut raw_ptr, size_in_bytes) != 0 {
                panic!("Недостаточно VRAM: Выделение {} байт (i32) провалено", size_in_bytes);
            }
        }
        CudaBuffer {
            raw_ptr,
            size_in_bytes,
            elements,
            is_owner: true,
        }
    }

    pub fn slice(&self, offset_bytes: usize, num_bytes: usize) -> Self {
        if self.size_in_bytes == 0 {
            panic!("Попытка сделать слайс от пустого CudaBuffer!");
        }
        assert!(offset_bytes + num_bytes <= self.size_in_bytes);
        unsafe {
            let sliced_ptr = (self.raw_ptr as *mut u8).add(offset_bytes) as *mut c_void;
            CudaBuffer {
                raw_ptr: sliced_ptr,
                size_in_bytes: num_bytes,
                elements: num_bytes,
                is_owner: false,
            }
        }
    }

    pub fn zero_out_async(&self, stream: &CudaStream) {
        unsafe {
            let _ = cudaMemsetAsync(self.raw_ptr, 0, self.size_in_bytes, stream.as_raw());
        }
    }

    pub fn len(&self) -> usize {
        self.elements
    }
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw_ptr
    }

    pub fn copy_from_host_slice<T: Copy>(&self, host_data: &[T], stream: &CudaStream) {
        let host_bytes = host_data.len() * size_of::<T>();
        assert!(host_bytes <= self.size_in_bytes);
        unsafe {
            let _ = cudaMemcpyAsync(
                self.raw_ptr,
                host_data.as_ptr() as *const c_void,
                host_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.as_raw(),
            );
        }
    }

    pub fn copy_to_host_slice<T: Copy>(&self, host_dst: &mut [T], stream: &CudaStream) {
        let host_bytes = host_dst.len() * size_of::<T>();
        assert!(host_bytes <= self.size_in_bytes);
        unsafe {
            let _ = cudaMemcpyAsync(
                host_dst.as_mut_ptr() as *mut c_void,
                self.raw_ptr,
                host_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream.as_raw(),
            );
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.is_owner && !self.raw_ptr.is_null() {
            unsafe {
                let _ = cudaFree(self.raw_ptr);
            }
        }
    }
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}
