use crate::utils::CudaBuffer;
use crate::models::kv_cache::KvCache;
use std::ffi::c_void;

unsafe extern "C" {
    // 1. Скалярное произведение Q * K
    fn launch_attention_scores(
        output_scores: *mut c_void,
        query: *const c_void,
        k_cache: *const c_void,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
    );

    // 2. Нормализация Softmax
    fn launch_softmax_attention(scores: *mut c_void, num_heads: i32, current_seq_len: i32);

    // 3. Сборка взвешенных векторов Value
    fn launch_attention_values(
        output: *mut c_void,
        probabilities: *const c_void,
        v_cache: *const c_void,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
    );
}

pub struct SelfAttention {
    pub w_query: CudaBuffer,
    pub w_key: CudaBuffer,
    pub w_value: CudaBuffer,
    pub w_out: CudaBuffer,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize, // Явно храним размерность головы в структуре
}

impl SelfAttention {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        // Жесткая проверка геометрии тензоров: hidden_size обязан делиться на num_heads нацело
        assert_eq!(
            hidden_size % num_heads, 0,
            "Критическая ошибка: hidden_size ({}) должен делиться на num_heads ({}) без остатка!",
            hidden_size, num_heads
        );

        let head_dim = hidden_size / num_heads; // Тот самый HEAD_DIM теперь рассчитывается и задействован!
        let kv_dim = num_kv_heads * head_dim;

        let w_query = CudaBuffer::new(hidden_size * hidden_size);
        let w_key = CudaBuffer::new(hidden_size * kv_dim);
        let w_value = CudaBuffer::new(hidden_size * kv_dim);
        let w_out = CudaBuffer::new(hidden_size * hidden_size);

        let init_weights = vec![0.1f32; hidden_size * hidden_size];
        w_query.copy_from_host(&init_weights);
        w_out.copy_from_host(&init_weights);

        let init_kv_weights = vec![0.1f32; hidden_size * kv_dim];
        w_key.copy_from_host(&init_kv_weights);
        w_value.copy_from_host(&init_kv_weights);

        SelfAttention {
            w_query,
            w_key,
            w_value,
            w_out,
            num_heads,
            num_kv_heads,
            head_dim, // Сохраняем задействованный параметр
        }
    }

    pub fn compute_attention_scores(&self, scores_output: &CudaBuffer, query_input: &CudaBuffer, kv_cache: &KvCache) {
        let current_seq_len = kv_cache.len() as i32;
        let n_heads = self.num_heads as i32;
        let n_kv_heads = self.num_kv_heads as i32;
        let h_dim = self.head_dim as i32; // Явно передаем задействованный head_dim на GPU!

        unsafe {
            launch_attention_scores(
                scores_output.as_raw_ptr(),
                query_input.as_raw_ptr(),
                kv_cache.k_cache.as_raw_ptr(),
                n_heads,
                n_kv_heads,
                h_dim,
                current_seq_len,
            );
        }
    }

    pub fn forward_softmax(&self, scores_buffer: &CudaBuffer, kv_cache: &KvCache) {
        let n_heads = self.num_heads as i32;
        let current_seq_len = kv_cache.len() as i32;

        unsafe {
            launch_softmax_attention(scores_buffer.as_raw_ptr(), n_heads, current_seq_len);
        }
    }

    pub fn forward_values(&self, attention_output: &CudaBuffer, probabilities: &CudaBuffer, kv_cache: &KvCache) {
        let n_heads = self.num_heads as i32;
        let n_kv_heads = self.num_kv_heads as i32;
        let h_dim = self.head_dim as i32; // Явно передаем задействованный head_dim на GPU!
        let current_seq_len = kv_cache.len() as i32;

        unsafe {
            launch_attention_values(
                attention_output.as_raw_ptr(),
                probabilities.as_raw_ptr(),
                kv_cache.v_cache.as_raw_ptr(),
                n_heads,
                n_kv_heads,
                h_dim,
                current_seq_len,
            );
        }
    }
}
