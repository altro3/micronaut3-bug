use crate::models::attention::SelfAttention;
use crate::models::kv_cache::KvCache;
use crate::models::layers::RmsNorm;
use crate::models::swiglu::SwiGlu;
use crate::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    // Импортируем наше новое ядро остаточной связи
    fn launch_residual(input_output: *mut c_void, residual_data: *const c_void, size: i32);
}

/// Полноценный промышленный блок модели TransformerBlock (Архитектура Qwen)
pub struct TransformerBlock {
    pub attn_norm: RmsNorm,       // Первый RmsNorm перед слоем внимания
    pub attention: SelfAttention, // Слой Grouped-Query внимания
    pub ffn_norm: RmsNorm,        // Второй RmsNorm перед скрытым слоем MLP
    pub mlp: SwiGlu,              // Скрытый MLP блок на базе активации SwiGLU

    hidden_size: usize,
}

impl TransformerBlock {
    /// Конструктор: инициализирует и связывает все внутренние математические слои блока
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        hidden_features: usize,
    ) -> Self {
        let attn_norm = RmsNorm::new(hidden_size);
        let attention = SelfAttention::new(hidden_size, num_heads, num_kv_heads);
        let ffn_norm = RmsNorm::new(hidden_size);
        let mlp = SwiGlu::new(hidden_size, hidden_features);

        TransformerBlock {
            attn_norm,
            attention,
            ffn_norm,
            mlp,
            hidden_size,
        }
    }

    /// Сквозной прямой проход (forward pass) одного блока модели Qwen.
    /// Вычисления происходят конвейером СТРОГО внутри VRAM, без участия ОЗУ хоста!
    pub fn forward(&self, x: &CudaBuffer, kv_cache: &mut KvCache, batch_size: usize) {
        let total_elements = (batch_size * self.hidden_size) as i32;

        // --- БЛОК 1: СЛОЙ ВНИМАНИЯ (ATTENTION) ---
        // Выделяем временные буферы под нормализацию и выход внимания
        let norm_hidden = CudaBuffer::new(batch_size * self.hidden_size);
        let attn_out = CudaBuffer::new(batch_size * self.hidden_size);

        // Шаг 1.1: Нормализуем входные данные
        self.attn_norm.forward(&norm_hidden, x, batch_size);

        // Шаг 1.2: Вычисляем сырые скоры и Softmax (для теста пока вызываем цепочку скоров)
        // В полноценной модели здесь также вычисляются Query/Key/Value проекции.
        let scores_buffer = CudaBuffer::new(self.attention.num_heads * kv_cache.len());
        let dummy_query = CudaBuffer::new(batch_size * self.hidden_size); // Временная заглушка для теста

        self.attention
            .compute_attention_scores(&scores_buffer, &dummy_query, kv_cache);
        self.attention.forward_softmax(&scores_buffer, kv_cache);
        self.attention
            .forward_values(&attn_out, &scores_buffer, kv_cache);

        // Шаг 1.3: Первая остаточная связь (x = x + attn_out) прямо на GPU
        unsafe {
            launch_residual(x.as_raw_ptr(), attn_out.as_raw_ptr(), total_elements);
        }

        // --- БЛОК 2: СКРЫТЫЙ СЛОЙ НЕЙРОНОВ (MLP / SwiGLU) ---
        let mlp_out = CudaBuffer::new(batch_size * self.hidden_size);

        // Шаг 2.1: Нормализуем данные перед MLP
        self.ffn_norm.forward(&norm_hidden, x, batch_size);

        // Шаг 2.2: Прогоняем данные через каскад трех матриц SwiGLU
        self.mlp.forward(&mlp_out, &norm_hidden, batch_size);

        // Шаг 2.3: Вторая остаточная связь (x = x + mlp_out) прямо на GPU
        unsafe {
            launch_residual(x.as_raw_ptr(), mlp_out.as_raw_ptr(), total_elements);
        }
    }
}
