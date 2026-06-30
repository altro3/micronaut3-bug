use crate::models::block::TransformerBlock;
use crate::models::kv_cache::KvCache;
use crate::models::linear::Linear;
use crate::models::rmsnorm::RmsNorm;
use crate::utils::cuda_stream::{cudaMemcpyAsync, CUDA_MEMCPY_DEVICE_TO_DEVICE};
use crate::utils::{CudaBuffer, CudaStream};

pub struct LlamaModel {
    pub token_embeddings: CudaBuffer,
    pub layers: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl LlamaModel {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        hidden_features: usize,
        num_layers: usize,
        vocab_size: usize,
        stream: &CudaStream,
    ) -> Self {
        let token_embeddings = CudaBuffer::new(vocab_size * hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(
                hidden_size,
                num_heads,
                num_kv_heads,
                hidden_features,
                stream,
            ));
        }

        let norm = RmsNorm::new(hidden_size, stream);
        let lm_head = Linear::new(hidden_size, vocab_size, stream);

        LlamaModel {
            token_embeddings,
            layers,
            norm,
            lm_head,
            hidden_size,
            vocab_size,
        }
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        kv_caches: &mut [KvCache],
        workspace: &CudaBuffer,
        logits_output: &CudaBuffer,
        stream: &CudaStream,
    ) {
        let num_tokens = token_ids.len();
        let size_hidden = num_tokens * self.hidden_size;

        let hidden_states = workspace.slice(0, size_hidden);
        let norm_out = workspace.slice(size_hidden, size_hidden);

        let layer_workspace_offset = size_hidden * 2;
        let layer_workspace_size = workspace.len() - layer_workspace_offset;
        let layer_workspace = workspace.slice(layer_workspace_offset, layer_workspace_size);

        unsafe {
            self.gather_embeddings_async(token_ids, &hidden_states, stream);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(
                &hidden_states,
                &kv_caches[i],
                num_tokens,
                &layer_workspace,
                stream,
            );
        }

        self.norm
            .forward(&norm_out, &hidden_states, num_tokens, stream);

        self.lm_head
            .forward(logits_output, &norm_out, num_tokens, stream);
    }

    unsafe fn gather_embeddings_async(
        &self,
        token_ids: &[u32],
        dst_hidden_states: &CudaBuffer,
        stream: &CudaStream,
    ) {
        let row_size_bytes = self.hidden_size * size_of::<f32>();

        unsafe {
            for (i, &token_id) in token_ids.iter().enumerate() {
                let valid_id = if (token_id as usize) < self.vocab_size {
                    token_id as usize
                } else {
                    0
                };

                let src_ptr =
                    (self.token_embeddings.as_raw_ptr() as *mut u8).add(valid_id * row_size_bytes);

                let dst_ptr = (dst_hidden_states.as_raw_ptr() as *mut u8).add(i * row_size_bytes);

                cudaMemcpyAsync(
                    dst_ptr as *mut libc::c_void,
                    src_ptr as *const libc::c_void,
                    row_size_bytes,
                    CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                );
            }
        }
    }

    pub fn required_workspace_elements(&self, batch_size: usize, max_seq_len: usize) -> usize {
        if self.layers.is_empty() {
            return 0;
        }
        let layer_req = self.layers[0].required_workspace_elements(batch_size, max_seq_len);
        let global_activations_overhead = (batch_size * max_seq_len * self.hidden_size) * 2;

        layer_req + global_activations_overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::kv_cache::KvCacheManager;

    #[test]
    fn test_full_llama_model_forward_pipeline_async() {
        const HIDDEN_SIZE: usize = 4;
        const NUM_HEADS: usize = 2;
        const NUM_KV_HEADS: usize = 1;
        const HIDDEN_FEATURES: usize = 4;
        const NUM_LAYERS: usize = 2;
        const VOCAB_SIZE: usize = 10;
        const MAX_SEQ_LEN: usize = 16;

        let stream = CudaStream::new();

        let model = LlamaModel::new(
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
            HIDDEN_FEATURES,
            NUM_LAYERS,
            VOCAB_SIZE,
            &stream,
        );

        let req_elements = model.required_workspace_elements(1, MAX_SEQ_LEN);
        let gpu_workspace = CudaBuffer::new(req_elements);
        let gpu_logits = CudaBuffer::new(VOCAB_SIZE);

        let kv_managers: Vec<KvCacheManager> = (0..NUM_LAYERS)
            .map(|_| KvCacheManager::new(MAX_SEQ_LEN, HIDDEN_SIZE))
            .collect();

        let mut kv_views: Vec<KvCache> = kv_managers.iter().map(|m| m.get_view()).collect();
        let input_tokens = vec![5u32];

        model.forward(
            &input_tokens,
            &mut kv_views,
            &gpu_workspace,
            &gpu_logits,
            &stream,
        );

        let mut host_logits = vec![0.0f32; VOCAB_SIZE];
        gpu_logits.copy_to_host_async(&mut host_logits, &stream);

        stream.synchronize();

        assert_eq!(host_logits.len(), VOCAB_SIZE);
        println!(
            "[ЮНИТ-ТЕСТ УСПЕШЕН] Граф полной LlamaModel собран, асинхронный forward работает без паник."
        );
    }
}
