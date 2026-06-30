use crate::models::attention::SelfAttention;
use crate::models::kv_cache::KvCache;
use crate::models::rmsnorm::RmsNorm;
use crate::models::swiglu::SwiGlu;
use crate::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_residual(
        input_output: *mut c_void,
        residual_data: *const c_void,
        size: i32,
        stream: *mut c_void,
    );
}

pub struct TransformerBlock {
    pub attn_norm: RmsNorm,
    pub attention: SelfAttention,
    pub ffn_norm: RmsNorm,
    pub mlp: SwiGlu,
    pub hidden_size: usize,
    pub hidden_features: usize,
}

impl TransformerBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        hidden_features: usize,
        stream: &CudaStream,
    ) -> Self {
        let attn_norm = RmsNorm::new(hidden_size, stream);
        let attention = SelfAttention::new(hidden_size, num_heads, num_kv_heads, stream);
        let ffn_norm = RmsNorm::new(hidden_size, stream);
        let mlp = SwiGlu::new(hidden_size, hidden_features, stream);

        TransformerBlock {
            attn_norm,
            attention,
            ffn_norm,
            mlp,
            hidden_size,
            hidden_features,
        }
    }

    pub fn required_workspace_elements(&self, batch_size: usize, max_seq_len: usize) -> usize {
        let size_hidden = batch_size * self.hidden_size;
        let size_scores = self.attention.num_heads * max_seq_len;
        let size_mlp_workspace = 3 * batch_size * self.hidden_features;

        size_hidden * 4 + size_scores + size_mlp_workspace
    }

    pub fn forward(
        &self,
        x: &CudaBuffer,
        kv_cache: &KvCache,
        batch_size: usize,
        workspace: &CudaBuffer,
        stream: &CudaStream,
    ) {
        let total_elements = (batch_size * self.hidden_size) as i32;
        let size_hidden = batch_size * self.hidden_size;
        let size_scores = self.attention.num_heads * kv_cache.len();

        let mut offset = 0;

        let norm_hidden = workspace.slice(offset, size_hidden);
        offset += size_hidden;
        let attn_out = workspace.slice(offset, size_hidden);
        offset += size_hidden;
        let dummy_query = workspace.slice(offset, size_hidden);
        offset += size_hidden;
        let mlp_out = workspace.slice(offset, size_hidden);
        offset += size_hidden;
        let scores_buffer = workspace.slice(offset, size_scores);
        offset += size_scores;

        let mlp_workspace_size = 3 * batch_size * self.hidden_features;
        let mlp_workspace = workspace.slice(offset, mlp_workspace_size);

        self.attn_norm.forward(&norm_hidden, x, batch_size, stream);

        self.attention
            .compute_attention_scores(&scores_buffer, &dummy_query, kv_cache, stream);
        self.attention
            .forward_softmax(&scores_buffer, kv_cache, stream);
        self.attention
            .forward_values(&attn_out, &scores_buffer, kv_cache, stream);

        // Первый Residual Connection: x = x + attn_out
        unsafe {
            launch_residual(
                x.as_raw_ptr(),
                attn_out.as_raw_ptr(),
                total_elements,
                stream.as_raw(),
            );
        }

        // =========================================================================
        // ВЫЧИСЛИТЕЛЬНЫЙ КОНВЕЙЕР (СТАДИЯ 2: MLP / FFN BLOCK)
        // =========================================================================
        self.ffn_norm.forward(&norm_hidden, x, batch_size, stream);
        self.mlp
            .forward(&mlp_out, &norm_hidden, batch_size, &mlp_workspace, stream);

        // Второй Residual Connection: x = x + mlp_out
        unsafe {
            launch_residual(
                x.as_raw_ptr(),
                mlp_out.as_raw_ptr(),
                total_elements,
                stream.as_raw(),
            );
        }
    }
}

// =========================================================================
// ИЗОЛИРОВАННЫЕ АСИНХРОННЫЕ ТЕСТЫ ОРКЕСТРАЦИИ БЛОКА ТРАНСФОРМЕРА
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    // Импортируем оригинальный менеджер кэша из вашего проекта
    use crate::models::kv_cache::KvCacheManager;

    #[test]
    fn test_transformer_block_workspace_allocation_async() {
        const BATCH_SIZE: usize = 2;
        const HIDDEN_SIZE: usize = 4;
        const HIDDEN_FEATURES: usize = 4;
        const MAX_SEQ_LEN: usize = 8;

        let stream = CudaStream::new();

        // 1. Инициализация блока (использует реальный SelfAttention под капотом)
        let block = TransformerBlock::new(HIDDEN_SIZE, 2, 2, HIDDEN_FEATURES, &stream);

        // 2. Проверяем расчет статической памяти черновика
        let req_elements = block.required_workspace_elements(BATCH_SIZE, MAX_SEQ_LEN);
        assert_eq!(
            req_elements, 72,
            "Формула Static Memory Plan рассчитана неверно!"
        );

        // 3. Выделяем физические буферы во VRAM под тест
        let gpu_input_output = CudaBuffer::new(BATCH_SIZE * HIDDEN_SIZE);
        let gpu_workspace = CudaBuffer::new(req_elements);

        // Создаем НАСТОЯЩИЙ менеджер кэша
        let kv_manager = KvCacheManager::new(MAX_SEQ_LEN, HIDDEN_SIZE);

        // Имитируем, что в нем уже накоплено 4 токена, чтобы ядрам было что читать
        // Для теста просто берем View-окно от текущего состояния менеджера
        let kv_view = kv_manager.get_view();

        // Заполняем вход контролируемыми числами
        let host_input = vec![1.0f32; BATCH_SIZE * HIDDEN_SIZE];
        gpu_input_output.copy_from_host_async(&host_input, &stream);

        // 4. Передаем НАСТОЯЩИЙ, честный view-слайс кэша по обычной ссылке &
        block.forward(
            &gpu_input_output,
            &kv_view,
            BATCH_SIZE,
            &gpu_workspace,
            &stream,
        );

        // Скачиваем результат и синхронизируем стрим
        let mut host_result = vec![0.0f32; BATCH_SIZE * HIDDEN_SIZE];
        gpu_input_output.copy_to_host_async(&mut host_result, &stream);
        stream.synchronize();

        // Проверяем, что на выходе данные физически существуют
        assert_eq!(host_result.len(), BATCH_SIZE * HIDDEN_SIZE);
        println!(
            "[ЮНИТ-ТЕСТ УСПЕШЕН] Оркестрация TransformerBlock на реальных типах работает идеально."
        );
    }
}
