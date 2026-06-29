use crate::models::attention::SelfAttention;
use crate::models::kv_cache::KvCache;
use crate::models::rmsnorm::RmsNorm;
use crate::models::swiglu::SwiGlu;
use crate::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_residual(input_output: *mut c_void, residual_data: *const c_void, size: i32);
}

pub struct TransformerBlock {
    pub attn_norm: RmsNorm,
    pub attention: SelfAttention,
    pub ffn_norm: RmsNorm,
    pub mlp: SwiGlu,
    hidden_size: usize,
}

impl TransformerBlock {
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

    pub fn forward(&self, x: &CudaBuffer, kv_cache: &mut KvCache, batch_size: usize) {
        let total_elements = (batch_size * self.hidden_size) as i32;

        let norm_hidden = CudaBuffer::new(batch_size * self.hidden_size);
        let attn_out = CudaBuffer::new(batch_size * self.hidden_size);

        self.attn_norm.forward(&norm_hidden, x, batch_size);

        let scores_buffer = CudaBuffer::new(self.attention.num_heads * kv_cache.len());
        let dummy_query = CudaBuffer::new(batch_size * self.hidden_size);

        self.attention
            .compute_attention_scores(&scores_buffer, &dummy_query, kv_cache);
        self.attention.forward_softmax(&scores_buffer, kv_cache);
        self.attention
            .forward_values(&attn_out, &scores_buffer, kv_cache);

        unsafe {
            launch_residual(x.as_raw_ptr(), attn_out.as_raw_ptr(), total_elements);
        }

        let mlp_out = CudaBuffer::new(batch_size * self.hidden_size);

        self.ffn_norm.forward(&norm_hidden, x, batch_size);
        self.mlp.forward(&mlp_out, &norm_hidden, batch_size);

        unsafe {
            launch_residual(x.as_raw_ptr(), mlp_out.as_raw_ptr(), total_elements);
        }
    }
}
