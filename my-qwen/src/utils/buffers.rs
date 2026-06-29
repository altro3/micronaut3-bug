use libc::c_int;
use std::ffi::c_void;

// 1. Объявляем низкоуровневые функции CUDA API, которые лежат в dll-библиотеке драйвера NVIDIA
unsafe extern "C" {
    // Выделяет память на GPU. Принимает указатель на указатель и размер в байтах
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;

    // Освобождает память на GPU по указателю
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;

    // Копирует данные (Host -> Device, Device -> Host, Device -> Device)
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
}

// Константы для направления копирования данных (стандарт CUDA API)
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// Безопасная RAII-обертка над непрерывным куском памяти во VRAM видеокарты
pub struct CudaBuffer {
    // Сырой указатель на память внутри GPU.
    // mut c_void в Rust — это аналог "void*" в Си/C++
    raw_ptr: *mut c_void,
    // Размер буфера в байтах
    size_in_bytes: usize,
}

impl CudaBuffer {
    /// Конструктор: выделяет память на видеокарте под заданное количество элементов float (f32)
    pub fn new(elements: usize) -> Self {
        let size_in_bytes = elements * std::mem::size_of::<f32>();
        let mut raw_ptr = std::ptr::null_mut();

        // Вызываем Си-функцию cudaMalloc. В Rust работа с сырыми указателями
        // считается небезопасной, поэтому мы обязаны обернуть её в блок unsafe.
        // Этим мы говорим компилятору: "Я беру ответственность за этот Си-код на себя"
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

    /// Копирует плоский вектор Vec<f32> из оперативной памяти (Host) на видеокарту (Device)
    pub fn copy_from_host(&self, host_data: &[f32]) {
        assert_eq!(
            host_data.len() * std::mem::size_of::<f32>(),
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

    /// Скачивает данные обратно с видеокарты (Device) в обычный вектор Rust (Host)
    pub fn copy_to_host(&self) -> Vec<f32> {
        let elements = self.size_in_bytes / std::mem::size_of::<f32>();
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

    /// Метод, позволяющий безопасно получить сырой указатель для передачи в наши CUDA .cu кернелы
    pub fn as_raw_ptr(&self) -> *mut c_void {
        self.raw_ptr
    }
}

// 2. РЕАЛИЗАЦИЯ ДЕСТРУКТОРА (Аналог финализатора в Java, но работающий со 100% гарантией)
// Интерфейс Drop вызывается автоматически компилятором Rust в тот момент,
// когда переменная типа CudaBuffer уничтожается.
impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.raw_ptr.is_null() {
            unsafe {
                let status = cudaFree(self.raw_ptr);
                if status != 0 {
                    // В деструкторах Rust не принято паниковать, поэтому просто выводим ошибку
                    eprintln!(
                        "Критическая ошибка при очистке VRAM: cudaFree вернул код {}",
                        status
                    );
                }
            }
        }
    }
}
