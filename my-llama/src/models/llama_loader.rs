use crate::models::llama_model::LlamaModel;
use crate::utils::safetensors::SafeTensorLoader;
use crate::utils::{CudaStream, PinnedHostBuffer};

impl LlamaModel {
    pub fn load_weights(
        &self,
        loader: &SafeTensorLoader,
        stream: &CudaStream,
    ) -> std::io::Result<()> {
        println!(
            "[MY-LLAMA] Начинается сквозная асинхронная заливка весов Qwen3.5-2B по PCIe DMA..."
        );

        loader.load_into_buffer(
            "model.language_model.embed_tokens.weight",
            &self.token_embeddings,
            stream,
        )?;

        let q_heads = 16;
        let kv_heads = 2;
        let total_heads = q_heads + kv_heads + kv_heads;

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{}.", i);

            loader.load_into_buffer(
                &format!("{}input_layernorm.weight", prefix),
                &layer.attn_norm.weight,
                stream,
            )?;
            loader.load_into_buffer(
                &format!("{}post_attention_layernorm.weight", prefix),
                &layer.ffn_norm.weight,
                stream,
            )?;

            let is_delta_net = loader
                .tensors
                .tensor(&format!("{}linear_attn.in_proj_qkv.weight", prefix))
                .is_ok();

            if is_delta_net {
                let qkv_name = format!("{}linear_attn.in_proj_qkv.weight", prefix);
                let qkv_view = loader.tensors.tensor(&qkv_name).unwrap();
                let qkv_raw_bytes = qkv_view.data();

                let total_bf16_elements = qkv_raw_bytes.len() / 2;
                let head_elements = total_bf16_elements / total_heads;

                let q_cap = layer.attention.projections.w_query.len();
                let k_cap = layer.attention.projections.w_key.len();

                let q_target_len = q_cap.min(q_heads * head_elements);
                let kv_target_len = k_cap.min(kv_heads * head_elements);

                let qkv_bf16_slice = unsafe {
                    std::slice::from_raw_parts(
                        qkv_raw_bytes.as_ptr() as *const u16,
                        total_bf16_elements,
                    )
                };

                let mut pinned_q = PinnedHostBuffer::new(q_target_len);
                let mut pinned_k = PinnedHostBuffer::new(kv_target_len);
                let mut pinned_v = PinnedHostBuffer::new(kv_target_len);

                let slice_q = pinned_q.as_slice_mut();
                let slice_k = pinned_k.as_slice_mut();
                let slice_v = pinned_v.as_slice_mut();

                let q_offset = 0;
                let k_offset = q_heads * head_elements;
                let v_offset = (q_heads + kv_heads) * head_elements;

                for j in 0..q_target_len {
                    slice_q[j] = f32::from_bits((qkv_bf16_slice[q_offset + j] as u32) << 16);
                }
                for j in 0..kv_target_len {
                    slice_k[j] = f32::from_bits((qkv_bf16_slice[k_offset + j] as u32) << 16);
                    slice_v[j] = f32::from_bits((qkv_bf16_slice[v_offset + j] as u32) << 16);
                }

                layer
                    .attention
                    .projections
                    .w_query
                    .copy_from_host_async(slice_q, stream);
                layer
                    .attention
                    .projections
                    .w_key
                    .copy_from_host_async(slice_k, stream);
                layer
                    .attention
                    .projections
                    .w_value
                    .copy_from_host_async(slice_v, stream);

                loader.load_into_buffer(
                    &format!("{}linear_attn.out_proj.weight", prefix),
                    &layer.attention.projections.w_out,
                    stream,
                )?;
            } else {
                loader.load_into_buffer(
                    &format!("{}self_attn.q_proj.weight", prefix),
                    &layer.attention.projections.w_query,
                    stream,
                )?;
                loader.load_into_buffer(
                    &format!("{}self_attn.k_proj.weight", prefix),
                    &layer.attention.projections.w_key,
                    stream,
                )?;
                loader.load_into_buffer(
                    &format!("{}self_attn.v_proj.weight", prefix),
                    &layer.attention.projections.w_value,
                    stream,
                )?;
                loader.load_into_buffer(
                    &format!("{}self_attn.o_proj.weight", prefix),
                    &layer.attention.projections.w_out,
                    stream,
                )?;
            }

            loader.load_into_buffer(
                &format!("{}mlp.gate_proj.weight", prefix),
                &layer.mlp.w_gate,
                stream,
            )?;
            loader.load_into_buffer(
                &format!("{}mlp.up_proj.weight", prefix),
                &layer.mlp.w_up,
                stream,
            )?;
            loader.load_into_buffer(
                &format!("{}mlp.down_proj.weight", prefix),
                &layer.mlp.w_down,
                stream,
            )?;
        }

        loader.load_into_buffer(
            "model.language_model.norm.weight",
            &self.norm.weight,
            stream,
        )?;

        let mut tied_weights = vec![0.0f32; self.token_embeddings.len()];
        self.token_embeddings
            .copy_to_host_async(&mut tied_weights, stream);
        stream.synchronize();
        self.lm_head
            .weight
            .data
            .copy_from_host_async(&tied_weights, stream);

        println!("[MY-LLAMA] Сквозная PCIe DMA загрузка и Weight Tying полностью завершены.");
        Ok(())
    }
}
