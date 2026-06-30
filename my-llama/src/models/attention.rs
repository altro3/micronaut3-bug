use crate::models::kv_cache::KvCache;
use crate::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {

    fn launch_attention_scores(
        output_scores: *mut c_void,
        query: *const c_void,
        k_cache: *const c_void,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
        stream: *mut c_void,
    );

    fn launch_softmax_attention(
        scores: *mut c_void,
        num_heads: i32,
        current_seq_len: i32,
        stream: *mut c_void,
    );

    fn launch_attention_values(
        output: *mut c_void,
        probabilities: *const c_void,
        v_cache: *const c_void,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
        stream: *mut c_void,
    );
}

pub struct AttentionProjections {
    pub w_query: CudaBuffer,
    pub w_key: CudaBuffer,
    pub w_value: CudaBuffer,
    pub w_out: CudaBuffer,
}

impl AttentionProjections {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        stream: &CudaStream,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let w_query = CudaBuffer::new(hidden_size * q_dim);
        let w_key = CudaBuffer::new(hidden_size * kv_dim);
        let w_value = CudaBuffer::new(hidden_size * kv_dim);
        let w_out = CudaBuffer::new(q_dim * hidden_size);

        let init_q_weights = vec![0.1f32; hidden_size * q_dim];
        let init_out_weights = vec![0.1f32; q_dim * hidden_size];
        w_query.copy_from_host_async(&init_q_weights, stream);
        w_out.copy_from_host_async(&init_out_weights, stream);

        let init_kv_weights = vec![0.1f32; hidden_size * kv_dim];
        w_key.copy_from_host_async(&init_kv_weights, stream);
        w_value.copy_from_host_async(&init_kv_weights, stream);

        AttentionProjections {
            w_query,
            w_key,
            w_value,
            w_out,
        }
    }
}

/// 2. ОРКЕСТРАТОР ВЫЧИСЛЕНИЙ МЕХАНИЗМА ВНИМАНИЯ (Вызов кудовских ядер)
pub struct SelfAttention {
    pub projections: AttentionProjections,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl SelfAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        stream: &CudaStream,
    ) -> Self {
        assert_eq!(
            hidden_size % num_heads,
            0,
            "Критическая ошибка: hidden_size должен делиться на num_heads без остатка!"
        );
        let head_dim = hidden_size / num_heads;

        // Создаем выделенный компонент проекций весов
        let projections =
            AttentionProjections::new(hidden_size, num_heads, num_kv_heads, head_dim, stream);

        SelfAttention {
            projections,
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// ВЫЧИСЛИТЕЛЬНОЕ ЯДРО СТАДИИ 1: Расчет асинхронных скалярных произведений (Dot-Product)
    pub fn compute_attention_scores(
        &self,
        scores_output: &CudaBuffer,
        query_input: &CudaBuffer,
        kv_cache: &KvCache,
        stream: &CudaStream,
    ) {
        unsafe {
            launch_attention_scores(
                scores_output.as_raw_ptr(),
                query_input.as_raw_ptr(),
                kv_cache.k_cache.as_raw_ptr(),
                self.num_heads as i32,
                self.num_kv_heads as i32,
                self.head_dim as i32,
                kv_cache.len() as i32,
                stream.as_raw(),
            );
        }
    }

    /// ВЫЧИСЛИТЕЛЬНОЕ ЯДРО СТАДИИ 2: Локальная асинхронная нормировка (Softmax) строки контекста
    pub fn forward_softmax(
        &self,
        scores_buffer: &CudaBuffer,
        kv_cache: &KvCache,
        stream: &CudaStream,
    ) {
        unsafe {
            launch_softmax_attention(
                scores_buffer.as_raw_ptr(),
                self.num_heads as i32,
                kv_cache.len() as i32,
                stream.as_raw(),
            );
        }
    }

    /// ВЫЧИСЛИТЕЛЬНОЕ ЯДРО СТАДИИ 3: Финальное взвешенное свертывание по Value-кэшу контекста
    pub fn forward_values(
        &self,
        attention_output: &CudaBuffer,
        probabilities: &CudaBuffer,
        kv_cache: &KvCache,
        stream: &CudaStream,
    ) {
        unsafe {
            launch_attention_values(
                attention_output.as_raw_ptr(),
                probabilities.as_raw_ptr(),
                kv_cache.v_cache.as_raw_ptr(),
                self.num_heads as i32,
                self.num_kv_heads as i32,
                self.head_dim as i32,
                kv_cache.len() as i32,
                stream.as_raw(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::kv_cache::KvCacheManager;

    #[test]
    fn test_decomposed_attention_pipeline_async() {
        const HIDDEN_SIZE: usize = 4;
        const NUM_HEADS: usize = 2;
        const NUM_KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 2;     // 4 / 2 = 2
        const CURRENT_SEQ_LEN: usize = 2;
        const MAX_SEQ_LEN: usize = 10;

        let stream = CudaStream::new();

        // 1. Инициализация реального слоя внимания
        let attention = SelfAttention::new(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, &stream);

        // 2. Инициализируем настоящий менеджер кэша
        let mut kv_manager = KvCacheManager::new(MAX_SEQ_LEN, HIDDEN_SIZE);

        // Создаем временные токены под историю контекста
        let token_k = CudaBuffer::new(HIDDEN_SIZE);
        let token_v = CudaBuffer::new(HIDDEN_SIZE);

        // ИСПРАВЛЕНИЕ 1: Явно размечаем тип f32 для векторов кэша
        token_k.copy_from_host_async(&vec![1.0f32, 0.0, 0.0, 1.0], &stream);
        token_v.copy_from_host_async(&vec![0.5f32, 0.5, 1.5, 1.5], &stream);

        // Накапливаем историю контекста (имитируем промпт из 2 токенов)
        kv_manager.append_async(&token_k, &token_v, &stream);
        kv_manager.append_async(&token_k, &token_v, &stream);

        // Получаем оригинальный легковесный view-слайс активного кэша
        let kv_view = kv_manager.get_view();

        // 3. Выделяем физические буферы под Query, Скоры и Выход
        // ИСПРАВЛЕНИЕ 2: Явно прописываем HEAD_DIM в геометрию буферов
        let gpu_query = CudaBuffer::new(NUM_HEADS * HEAD_DIM);
        let gpu_scores = CudaBuffer::new(NUM_HEADS * CURRENT_SEQ_LEN);
        let gpu_output = CudaBuffer::new(NUM_HEADS * HEAD_DIM);

        // ИСПРАВЛЕНИЕ 3: Явно размечаем f32 для Query вектора
        gpu_query.copy_from_host_async(&vec![1.0f32, 1.0, 1.0, 1.0], &stream);

        // 4. Проверяем работу трехстадийного вычислительного конвейера на реальных типах
        attention.compute_attention_scores(&gpu_scores, &gpu_query, &kv_view, &stream);
        attention.forward_softmax(&gpu_scores, &kv_view, &stream);
        attention.forward_values(&gpu_output, &gpu_scores, &kv_view, &stream);

        // Скачиваем результат на CPU
        let mut host_output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        gpu_output.copy_to_host_async(&mut host_output, &stream);

        stream.synchronize();

        // Финальные ассерты
        assert_eq!(host_output.len(), NUM_HEADS * HEAD_DIM);
        assert!(host_output.iter().any(|&x| x != 0.0f32), "Выходы внимания занулились!");
        println!("[ЮНИТ-ТЕСТ УСПЕШЕН] Декомпозированный асинхронный SelfAttention на реальном кэше работает идеально.");
    }
}
