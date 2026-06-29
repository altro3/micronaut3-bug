use crate::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_matmul(C: *mut c_void, A: *const c_void, B: *const c_void, M: i32, N: i32, K: i32);
    fn launch_swish_glu(output: *mut c_void, gate_input: *const c_void, up_input: *const c_void, size: i32);
}

/// Полноценный промышленный слой SwiGLU MLP-блока (Архитектура Qwen)
pub struct SwiGlu {
    pub w_gate: CudaBuffer, // Матрица Врата
    pub w_up: CudaBuffer,   // Матрица Ап
    pub w_down: CudaBuffer, // Матрица Даун

    in_features: usize,
    hidden_features: usize,
}

impl SwiGlu {
    /// Конструктор: выделяет VRAM под 3 матрицы весов.
    /// В LLM размерность hidden_features (скрытый слой MLP) обычно равна примерно 8/3 от размерности модели.
    pub fn new(in_features: usize, hidden_features: usize) -> Self {
        let w_gate = CudaBuffer::new(in_features * hidden_features);
        let w_up = CudaBuffer::new(in_features * hidden_features);
        let w_down = CudaBuffer::new(hidden_features * in_features);

        // Инициализируем веса тестовыми значениями (например, 0.5f32) для верификации математики
        let init_w = vec![0.5f32; in_features * hidden_features];
        let init_w_down = vec![0.2f32; hidden_features * in_features];

        w_gate.copy_from_host(&init_w);
        w_up.copy_from_host(&init_w);
        w_down.copy_from_host(&init_w_down);

        SwiGlu {
            w_gate,
            w_up,
            w_down,
            in_features,
            hidden_features,
        }
    }

    /// Прямой проход SwiGLU. Избегает промежуточных скачиваний в RAM!
    pub fn forward(&self, output: &CudaBuffer, input: &CudaBuffer, batch_size: usize) {
        let m = batch_size as i32;
        let n = self.hidden_features as i32;
        let k = self.in_features as i32;

        // Выделяем временные буферы во VRAM для промежуточных результатов этапа 1
        // Они автоматически удалятся (cudaFree) в конце этого метода!
        let h_gate = CudaBuffer::new(batch_size * self.hidden_features);
        let h_up = CudaBuffer::new(batch_size * self.hidden_features);
        let h_fused = CudaBuffer::new(batch_size * self.hidden_features);

        unsafe {
            // Этап 1: Параллельное матричное умножение входных данных на Gate и Up веса
            launch_matmul(h_gate.as_raw_ptr(), input.as_raw_ptr(), self.w_gate.as_raw_ptr(), m, n, k);
            launch_matmul(h_up.as_raw_ptr(), input.as_raw_ptr(), self.w_up.as_raw_ptr(), m, n, k);

            // Этап 2: Fused-кернел активации SiLU и схлопывания во VRAM
            let total_fused_elements = (batch_size * self.hidden_features) as i32;
            launch_swish_glu(h_fused.as_raw_ptr(), h_gate.as_raw_ptr(), h_up.as_raw_ptr(), total_fused_elements);

            // Этап 3: Финальное сжатие размерности матричным умножением на W_down
            let n_down = self.in_features as i32;
            let k_down = self.hidden_features as i32;
            launch_matmul(output.as_raw_ptr(), h_fused.as_raw_ptr(), self.w_down.as_raw_ptr(), m, n_down, k_down);
        }
    }
}
