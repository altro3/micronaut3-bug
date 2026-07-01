use crate::models::compiler::ops::Op;
use crate::models::types::TensorView;
use crate::utils::parameter::DataType;

#[derive(Debug, Clone)]
pub struct WeightSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

pub struct CompiledPipeline {
    pub pipeline: Vec<Op>,
    pub max_arena_bytes: usize,
    pub weight_specs: Vec<WeightSpec>,
    pub logits_tensor: TensorView,
    pub targets_offset: usize,
}

pub struct LlamaGraphCompiler;

impl LlamaGraphCompiler {
    pub fn compile(
        num_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        rms_norm_eps: f32,
        dtype: DataType,
        batch_size: usize,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        is_training: bool,
    ) -> CompiledPipeline {
        let mut pipeline = Vec::new();
        let mut weight_specs = Vec::new();
        let mut current_weight_idx = 0;

        let element_size = dtype.element_size();

        let hidden_state_bytes = batch_size * hidden_size * element_size;
        let intermediate_bytes = batch_size * intermediate_size * element_size;
        let logits_bytes = batch_size * vocab_size * element_size;

        let input_tokens_offset = 0;
        let input_tokens_bytes = batch_size * 4;

        let targets_offset = input_tokens_offset + input_tokens_bytes;
        let targets_bytes = if is_training { batch_size * 4 } else { 0 };

        let residual_stream_offset = targets_offset + targets_bytes;
        let mut current_arena_offset = residual_stream_offset + hidden_state_bytes;

        let temporary_a_offset = residual_stream_offset + hidden_state_bytes;
        let temporary_b_offset = temporary_a_offset + intermediate_bytes;
        let temporary_c_offset = temporary_b_offset + intermediate_bytes;

        weight_specs.push(WeightSpec {
            name: "model.embed_tokens.weight".to_string(),
            shape: vec![vocab_size, hidden_size],
        });

        let residual_stream_view = TensorView {
            offset: residual_stream_offset,
            bytes: hidden_state_bytes,
            batch_size: batch_size as i32,
            out_features: hidden_size as i32,
            in_features: hidden_size as i32,
        };

        pipeline.push(Op::Embeddings {
            input_id: input_tokens_offset,
            weight_idx: current_weight_idx,
            output: residual_stream_view,
        });
        current_weight_idx += 1;

        for layer_idx in 0..num_layers {
            let (layer_a_offset, layer_b_offset, layer_c_offset) = if is_training {
                let a = current_arena_offset;
                let b = a + hidden_state_bytes;
                let c = b + intermediate_bytes;
                current_arena_offset = c + intermediate_bytes;
                (a, b, c)
            } else {
                (temporary_a_offset, temporary_b_offset, temporary_c_offset)
            };

            weight_specs.push(WeightSpec {
                name: format!("model.layers.{}.input_layernorm.weight", layer_idx),
                shape: vec![hidden_size],
            });

            let attn_norm_out_view = TensorView {
                offset: layer_a_offset,
                bytes: hidden_state_bytes,
                batch_size: batch_size as i32,
                out_features: hidden_size as i32,
                in_features: hidden_size as i32,
            };

            pipeline.push(Op::RmsNorm {
                input: residual_stream_view,
                weight_idx: current_weight_idx,
                output: attn_norm_out_view,
                eps: rms_norm_eps,
            });
            current_weight_idx += 1;

            let attn_out_view = TensorView {
                offset: layer_b_offset,
                bytes: hidden_state_bytes,
                batch_size: batch_size as i32,
                out_features: hidden_size as i32,
                in_features: hidden_size as i32,
            };

            pipeline.push(Op::RoPEAndAttention {
                input: attn_norm_out_view,
                layer_idx,
                output: attn_out_view,
                num_heads,
                num_kv_heads,
                head_dim,
            });

            pipeline.push(Op::Add {
                a: residual_stream_view,
                b: attn_out_view,
                out: residual_stream_view,
            });

            weight_specs.push(WeightSpec {
                name: format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
                shape: vec![hidden_size],
            });

            let mlp_norm_out_view = TensorView {
                offset: layer_a_offset,
                bytes: hidden_state_bytes,
                batch_size: batch_size as i32,
                out_features: hidden_size as i32,
                in_features: hidden_size as i32,
            };

            pipeline.push(Op::RmsNorm {
                input: residual_stream_view,
                weight_idx: current_weight_idx,
                output: mlp_norm_out_view,
                eps: rms_norm_eps,
            });
            current_weight_idx += 1;

            weight_specs.push(WeightSpec {
                name: format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
                shape: vec![intermediate_size, hidden_size],
            });

            let gate_out_view = TensorView {
                offset: layer_b_offset,
                bytes: intermediate_bytes,
                batch_size: batch_size as i32,
                out_features: intermediate_size as i32,
                in_features: hidden_size as i32,
            };

            pipeline.push(Op::MatMul {
                input: mlp_norm_out_view,
                weight_idx: current_weight_idx,
                output: gate_out_view,
            });
            current_weight_idx += 1;

            weight_specs.push(WeightSpec {
                name: format!("model.layers.{}.mlp.up_proj.weight", layer_idx),
                shape: vec![intermediate_size, hidden_size],
            });

            let up_out_view = TensorView {
                offset: layer_c_offset,
                bytes: intermediate_bytes,
                batch_size: batch_size as i32,
                out_features: intermediate_size as i32,
                in_features: hidden_size as i32,
            };

            pipeline.push(Op::MatMul {
                input: mlp_norm_out_view,
                weight_idx: current_weight_idx,
                output: up_out_view,
            });
            current_weight_idx += 1;

            let swiglu_out_view = TensorView {
                offset: layer_a_offset,
                bytes: hidden_state_bytes,
                batch_size: batch_size as i32,
                out_features: hidden_size as i32,
                in_features: hidden_size as i32,
            };

            pipeline.push(Op::SwiGlu {
                input: gate_out_view,
                output: swiglu_out_view,
            });

            pipeline.push(Op::Add {
                a: residual_stream_view,
                b: swiglu_out_view,
                out: residual_stream_view,
            });
        }

        weight_specs.push(WeightSpec {
            name: "model.norm.weight".to_string(),
            shape: vec![hidden_size],
        });

        let (final_norm_out_offset, logits_offset) = if is_training {
            let norm_off = current_arena_offset;
            let logits_off = norm_off + hidden_state_bytes;
            (norm_off, logits_off)
        } else {
            (temporary_a_offset, temporary_c_offset + hidden_state_bytes)
        };

        let final_norm_out_view = TensorView {
            offset: final_norm_out_offset,
            bytes: hidden_state_bytes,
            batch_size: batch_size as i32,
            out_features: hidden_size as i32,
            in_features: hidden_size as i32,
        };

        pipeline.push(Op::RmsNorm {
            input: residual_stream_view,
            weight_idx: current_weight_idx,
            output: final_norm_out_view,
            eps: rms_norm_eps,
        });
        current_weight_idx += 1;

        weight_specs.push(WeightSpec {
            name: "lm_head.weight".to_string(),
            shape: vec![vocab_size, hidden_size],
        });

        let logits_view = TensorView {
            offset: logits_offset,
            bytes: logits_bytes,
            batch_size: batch_size as i32,
            out_features: vocab_size as i32,
            in_features: hidden_size as i32,
        };

        pipeline.push(Op::MatMul {
            input: final_norm_out_view,
            weight_idx: current_weight_idx,
            output: logits_view,
        });

        let total_arena_bytes = logits_offset + logits_bytes;

        CompiledPipeline {
            pipeline,
            max_arena_bytes: total_arena_bytes,
            weight_specs,
            logits_tensor: logits_view,
            targets_offset,
        }
    }
}
