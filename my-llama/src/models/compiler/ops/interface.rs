use crate::cuda::CudaStream;
use crate::models::types::TensorView;
use crate::utils::parameter::Parameter;
use std::ffi::c_void;

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Embeddings { input_id: usize, weight_idx: usize, output: TensorView },
    RmsNorm { input: TensorView, weight_idx: usize, output: TensorView, eps: f32 },
    MatMul { input: TensorView, weight_idx: usize, output: TensorView },
    Add { a: TensorView, b: TensorView, out: TensorView },
    SwiGlu { input: TensorView, output: TensorView },
    RoPEAndAttention {
        input: TensorView, layer_idx: usize, output: TensorView,
        num_heads: i32, num_kv_heads: i32, head_dim: i32,
    },
    MambaScan { input: TensorView, state_weight_idx: usize, output: TensorView },
    FusedCrossEntropy { logits: TensorView, targets_id: usize, d_logits: TensorView, losses: TensorView },
    MatMulBackward { d_output: TensorView, input: TensorView, weight_idx: usize, d_input: TensorView },
    AdamWStep {
        weight_idx: usize, lr: f32, beta1: f32, beta2: f32, eps: f32,
        weight_decay: f32, step: f32,
    },
}

impl Op {
    #[inline(always)]
    pub unsafe fn execute(&self, arena_ptr: *mut c_void, weights: &[Parameter], stream: &CudaStream) {
        use super::{attention, backward, forward_basic, optimizer, swiglu};

        match *self {
            Op::Embeddings { .. } | Op::RmsNorm { .. } | Op::MatMul { .. } | Op::Add { .. } => {
                unsafe { forward_basic::dispatch_forward(self, arena_ptr, weights, stream); }
            }
            Op::SwiGlu { input, output } => {
                unsafe { swiglu::dispatch_swiglu(arena_ptr, input, output, stream); }
            }
            Op::RoPEAndAttention { input, output, num_heads, num_kv_heads, head_dim, .. } => {
                unsafe { attention::dispatch_attention(arena_ptr, input, output, num_heads, num_kv_heads, head_dim, stream); }
            }
            Op::FusedCrossEntropy { .. } | Op::MatMulBackward { .. } => {
                unsafe { backward::dispatch_backward(self, arena_ptr, weights, stream); }
            }
            Op::AdamWStep { .. } => {
                unsafe { optimizer::dispatch_optimizer(self, weights, stream); }
            }
            Op::MambaScan { .. } => {
                unimplemented!("MambaSelectiveScan будет интегрирован в будущем");
            }
        }
    }
}
