use libc::c_int;
use std::ffi::c_void;

unsafe extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;

    fn cudaFree(dev_ptr: *mut c_void) -> c_int;

    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

pub struct CudaBuffer {
    raw_ptr: *mut c_void,
    size_in_bytes: usize,
}

impl CudaBuffer {
    pub fn new(elements: usize) -> Self {
        let size_in_bytes = elements * size_of::<f32>();
        let mut raw_ptr = std::ptr::null_mut();

        unsafe {
            let status = cudaMalloc(&mut raw_ptr, size_in_bytes);
            if status != 0 {
                panic!(
                    "Ошибка CUDA: Не удалось выделить {} байт во VRAM. Код ошибки: {}",
                    size_in_bytes, status
                );
            }
        }

        CudaBuffer {
            raw_ptr,
            size_in_bytes,
        }
    }

    pub fn new_int(elements: usize) -> Self {
        let size_in_bytes = elements * size_of::<i32>();
        let mut raw_ptr = std::ptr::null_mut();

        unsafe {
            let status = cudaMalloc(&mut raw_ptr, size_in_bytes);
            if status != 0 {
                panic!(
                    "Ошибка CUDA: Не удалось выделить {} байт (i32) во VRAM. Код ошибки: {}",
                    size_in_bytes, status
                );
            }
        }

        CudaBuffer {
            raw_ptr,
            size_in_bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.size_in_bytes / size_of::<f32>()
    }

    pub fn copy_from_host_int(&self, host_data: &[i32]) {
        assert_eq!(
            host_data.len() * size_of::<i32>(),
            self.size_in_bytes,
            "Размер входящих i32 данных не совпадает с размером буфера GPU!"
        );

        unsafe {
            let status = cudaMemcpy(
                self.raw_ptr,
                host_data.as_ptr() as *const c_void,
                self.size_in_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if status != 0 {
                panic!(
                    "Ошибка CUDA: Не удалось скопировать i32 данные на GPU. Код: {}",
                    status
                );
            }
        }
    }

    pub fn copy_from_host(&self, host_data: &[f32]) {
        assert_eq!(
            host_data.len() * size_of::<f32>(),
            self.size_in_bytes,
            "Размер входящих данных не совпадает с размером буфера GPU!"
        );

        unsafe {
            let status = cudaMemcpy(
                self.raw_ptr,
                host_data.as_ptr() as *const c_void,
                self.size_in_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if status != 0 {
                panic!(
                    "Ошибка CUDA: Не удалось скопировать данные на GPU. Код: {}",
                    status
                );
            }
        }
    }

    pub fn copy_to_host(&self) -> Vec<f32> {
        let elements = self.size_in_bytes / size_of::<f32>();
        // Создаем пустой вектор нужного размера, заполненный нулями
        let mut host_data = vec![0.0f32; elements];

        unsafe {
            let status = cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.raw_ptr,
                self.size_in_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            if status != 0 {
                panic!(
                    "Ошибка CUDA: Не удалось скачать данные с GPU. Код: {}",
                    status
                );
            }
        }
        host_data
    }

    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw_ptr
    }

    pub fn new_int_scalar() -> *mut c_void {
        let mut raw_ptr = std::ptr::null_mut();
        unsafe {
            let status = cudaMalloc(&mut raw_ptr, size_of::<i32>());
            if status != 0 {
                panic!("CUDA malloc for int scalar failed!");
            }
        }
        raw_ptr
    }

    pub fn copy_int_to_host(device_ptr: *mut c_void) -> i32 {
        let mut host_val: i32 = 0;
        unsafe {
            cudaMemcpy(
                &mut host_val as *mut i32 as *mut c_void,
                device_ptr,
                size_of::<i32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
        }
        host_val
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.raw_ptr.is_null() {
            unsafe {
                let status = cudaFree(self.raw_ptr);
                if status != 0 {
                    eprintln!(
                        "Критическая ошибка при очистке VRAM: cudaFree вернул код {}",
                        status
                    );
                }
            }
        }
    }
}
