use crate::utils::CudaBuffer;
use std::ffi::c_void;

// 1. Объявляем нашу Си-функцию, которую мы только что написали в softmax.cu.
// Используем спецификацию Rust Edition 2024 (unsafe extern).
unsafe extern "C" {
    fn launch_rms_norm(
        output: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        batch_size: i32,
        hidden_size: i32,
        epsilon: f32,
    );
}

pub struct RmsNorm {
    // Вектор обучаемых весов (гамма), живущий строго во VRAM
    pub weight: CudaBuffer,
    // Константа защиты от деления на ноль
    pub epsilon: f32,
    // Размерность скрытого слоя модели (hidden_size)
    pub hidden_size: usize,
}

impl RmsNorm {
    /// Конструктор: выделяет память под веса и заполняет их дефолтными единицами
    pub fn new(hidden_size: usize) -> Self {
        let weight_buffer = CudaBuffer::new(hidden_size);
        let initial_weights = vec![1.0f32; hidden_size];
        weight_buffer.copy_from_host(&initial_weights);

        RmsNorm {
            weight: weight_buffer,
            epsilon: 1e-5, // Стандарт для Qwen
            hidden_size,
        }
    }

    /// Прямой проход вычислений (Forward pass) слоя RMSNorm.
    /// Принимает входной буфер GPU и записывает результат в выходной буфер GPU.
    pub fn forward(&self, output: &CudaBuffer, input: &CudaBuffer, batch_size: usize) {
        // Вычисляем, сколько всего токенов (строк) сейчас обрабатывается
        let b_size = batch_size as i32;
        let h_size = self.hidden_size as i32;

        // Передаем управление видеокарте через FFI-мост.
        // Так как мы работаем с сырыми указателями, этот вызов обязан быть обернут в unsafe.
        unsafe {
            launch_rms_norm(
                output.as_raw_ptr(),  // Куда писать результат во VRAM
                input.as_raw_ptr(),   // Откуда брать исходные данные во VRAM
                self.weight.as_raw_ptr(), // Где лежат веса слоя во VRAM
                b_size,
                h_size,
                self.epsilon,
            );
        }
    }
}
