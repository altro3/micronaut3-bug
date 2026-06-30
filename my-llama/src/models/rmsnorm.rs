use crate::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_rms_norm(
        output: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        batch_size: i32,
        hidden_size: i32,
        epsilon: f32,
        stream: *mut c_void,
    );
}

pub struct RmsNorm {
    pub weight: CudaBuffer,
    pub hidden_size: usize,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, stream: &CudaStream) -> Self {
        let weight = CudaBuffer::new(hidden_size);

        let init_weights = vec![1.0f32; hidden_size];
        weight.copy_from_host_async(&init_weights, stream);

        RmsNorm {
            weight,
            hidden_size,
        }
    }

    pub fn forward(
        &self,
        output: &CudaBuffer,
        input: &CudaBuffer,
        batch_size: usize,
        stream: &CudaStream,
    ) {
        unsafe {
            launch_rms_norm(
                output.as_raw_ptr(),
                input.as_raw_ptr(),
                self.weight.as_raw_ptr(),
                batch_size as i32,
                self.hidden_size as i32,
                1e-5f32,
                stream.as_raw(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_forward_math_async() {
        const BATCH_SIZE: usize = 1;
        const HIDDEN_SIZE: usize = 3;

        let stream = CudaStream::new();
        let norm = RmsNorm::new(HIDDEN_SIZE, &stream);

        let gpu_input = CudaBuffer::new(BATCH_SIZE * HIDDEN_SIZE);
        let gpu_output = CudaBuffer::new(BATCH_SIZE * HIDDEN_SIZE);

        // ИСПРАВЛЕНИЕ 1: Явно указываем тип f32 для входных данных
        gpu_input.copy_from_host_async(&vec![1.0f32, 2.0, 3.0], &stream);

        // Запускаем асинхронный RMSNorm
        norm.forward(&gpu_output, &gpu_input, BATCH_SIZE, &stream);

        let mut host_output = vec![0.0f32; BATCH_SIZE * HIDDEN_SIZE];
        gpu_output.copy_to_host_async(&mut host_output, &stream);

        // Синхронизация перед ассертами
        stream.synchronize();

        // Ожидаемый эталонный ответ на CPU
        let expected = vec![0.46291f32, 0.92582, 1.38873];

        // ИСПРАВЛЕНИЕ 2: Проверяем данные строго поэлементно с дельтой,
        // убирая ошибочный assert_eq! массивов целиком
        for i in 0..HIDDEN_SIZE {
            assert!(
                (host_output[i] - expected[i]).abs() < 1e-4,
                "Ошибка математики RMSNorm на индексе {}: ожидали {}, получили {}",
                i,
                expected[i],
                host_output[i]
            );
        }
        println!("[ЮНИТ-ТЕСТ УСПЕШЕН] Асинхронный RMSNorm считает абсолютно точно.");
    }
}
