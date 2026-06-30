use crate::utils::cuda_stream::{
    cudaFree, cudaMalloc, cudaMemcpyAsync, cudaMemsetAsync,
    CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE,
};
use std::ffi::c_void;
use crate::utils::CudaStream;

pub struct CudaBuffer {
    raw_ptr: *mut c_void,
    size_in_bytes: usize,
    elements: usize,
    is_owner: bool,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

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

    pub fn slice(&self, offset_elements: usize, num_elements: usize) -> Self {
        if self.elements == 0 {
            panic!("Попытка сделать слайс от пустого CudaBuffer!");
        }
        let element_size = self.size_in_bytes / self.elements;
        let offset_bytes = offset_elements * element_size;
        let size_in_bytes = num_elements * element_size;

        assert!(
            offset_bytes + size_in_bytes <= self.size_in_bytes,
            "КРИТИЧЕСКИЙ ВЫХОД ЗА ГРАНИЦЫ VRAM БУФЕРА! Доступ к байтам {}..{}, доступно всего {}",
            offset_bytes, offset_bytes + size_in_bytes, self.size_in_bytes
        );

        unsafe {
            let sliced_ptr = (self.raw_ptr as *mut u8).add(offset_bytes) as *mut c_void;
            CudaBuffer {
                raw_ptr: sliced_ptr,
                size_in_bytes,
                elements: num_elements,
                is_owner: false,
            }
        }
    }

    pub fn zero_out_async(&self, stream: &CudaStream) {
        unsafe {
            cudaMemsetAsync(self.raw_ptr, 0, self.size_in_bytes, stream.as_raw());
        }
    }

    pub fn len(&self) -> usize {
        self.elements
    }

    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw_ptr
    }

    pub fn copy_from_host_async<T: Copy>(&self, host_data: &[T], stream: &CudaStream) {
        assert_eq!(
            host_data.len() * size_of::<T>(),
            self.size_in_bytes,
            "Ошибка DMA: Размер передаваемых данных в байтах не совпадает с буфером GPU!"
        );

        unsafe {
            cudaMemcpyAsync(
                self.raw_ptr,
                host_data.as_ptr() as *const c_void,
                self.size_in_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.as_raw(),
            );
        }
    }

    pub fn copy_to_host_async<T: Copy>(&self, host_dst: &mut [T], stream: &CudaStream) {
        assert_eq!(
            host_dst.len() * size_of::<T>(),
            self.size_in_bytes,
            "Ошибка DMA: Размер хост-массива приемника не совпадает с буфером GPU!"
        );

        unsafe {
            cudaMemcpyAsync(
                host_dst.as_mut_ptr() as *mut c_void,
                self.raw_ptr,
                self.size_in_bytes,
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
                cudaFree(self.raw_ptr);
            }
        }
    }
}
