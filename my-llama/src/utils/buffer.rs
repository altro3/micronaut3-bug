use libc::c_int;
use std::ffi::c_void;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaMemsetAsync(
        dev_ptr: *mut c_void,
        value: c_int,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;

    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;

    pub fn cudaStreamCreate(stream: *mut *mut c_void) -> c_int;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const _CUDA_HOST_ALLOC_DEFAULT: u32 = 0x00;

pub struct CudaStream {
    raw: *mut c_void,
}

impl CudaStream {
    pub fn new() -> Self {
        let mut raw = std::ptr::null_mut();
        unsafe {
            let status = cudaStreamCreate(&mut raw);
            if status != 0 {
                panic!("Не удалось создать CUDA Stream");
            }
        }
        CudaStream { raw }
    }

    pub fn synchronize(&self) {
        unsafe {
            cudaStreamSynchronize(self.raw);
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
                panic!("Ошибка CUDA: Выделение {} байт провалено", size_in_bytes);
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
                panic!("Ошибка CUDA: Выделение {} байт провалено", size_in_bytes);
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
        let element_size = self.size_in_bytes / self.elements;
        let offset_bytes = offset_elements * element_size;
        let size_in_bytes = num_elements * element_size;
        assert!(offset_bytes + size_in_bytes <= self.size_in_bytes);

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
            "Размер передаваемых данных в байтах не совпадает с размером буфера GPU!"
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
            "Размер массива-приемника в байтах не совпадает с размером буфера GPU!"
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

pub struct PinnedHostBuffer {
    raw_ptr: *mut c_void,
    elements: usize,
}

impl PinnedHostBuffer {
    pub fn new(elements: usize) -> Self {
        let size_in_bytes = elements * size_of::<f32>();
        let mut raw_ptr = std::ptr::null_mut();
        unsafe {
            if cudaHostAlloc(&mut raw_ptr, size_in_bytes, _CUDA_HOST_ALLOC_DEFAULT) != 0 {
                panic!("Не удалось выделить Pinned Memory на CPU");
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
    fn test_cuda_buffer_f32_lifecycle_async() {
        let elements = 4;
        let input_data = vec![1.1f32, -2.2f32, 3.3f32, 0.0f32];
        let mut output_data = vec![0.0f32; elements];

        let stream = CudaStream::new();
        let buffer = CudaBuffer::new(elements);
        assert_eq!(buffer.len(), elements);

        buffer.copy_from_host_async(&input_data, &stream);
        buffer.copy_to_host_async(&mut output_data, &stream);

        stream.synchronize();

        assert_eq!(input_data.len(), output_data.len());
        for i in 0..elements {
            assert!((input_data[i] - output_data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cuda_buffer_i32_lifecycle_async() {
        let elements = 3;
        let input_targets = vec![12, 42, 99];
        let mut output_targets = vec![0; elements];

        let stream = CudaStream::new();
        let buffer = CudaBuffer::new_int(elements);

        assert_eq!(buffer.len(), elements);

        buffer.copy_from_host_async(&input_targets, &stream);
        buffer.copy_to_host_async(&mut output_targets, &stream);

        stream.synchronize();

        assert_eq!(
            output_targets[1], 42,
            "Данные i32 прочитаны неверно, ожидали 42, получили {}",
            output_targets[1]
        );
        assert_eq!(
            output_targets, input_targets,
            "Массив i32 повредился при передаче!"
        );
    }

    #[test]
    fn test_cuda_buffer_slicing_async() {
        let parent = CudaBuffer::new(6);
        let host_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut host_slice_check = vec![0.0f32; 3];

        let stream = CudaStream::new();
        parent.copy_from_host_async(&host_data, &stream);

        let child_slice = parent.slice(2, 3);

        assert_eq!(child_slice.len(), 3, "Размер слайса должен быть равен 3");
        assert_eq!(
            child_slice.is_owner, false,
            "Слайс не должен владеть физической памятью"
        );

        child_slice.copy_to_host_async(&mut host_slice_check, &stream);
        stream.synchronize();

        assert_eq!(
            host_slice_check,
            vec![30.0, 40.0, 50.0],
            "Слайс прочитал неверное смещение памяти!"
        );
    }

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

        assert!(
            !pinned_buf.as_ptr().is_null(),
            "Указатель на Pinned память равен NULL"
        );
    }

    #[test]
    #[should_panic(expected = "Размер передаваемых данных в байтах не совпадает")]
    fn test_cuda_buffer_bounds_check_async() {
        let buffer = CudaBuffer::new(10);
        let invalid_data = vec![1.0f32; 5];
        let stream = CudaStream::new();

        buffer.copy_from_host_async(&invalid_data, &stream);
    }
}
