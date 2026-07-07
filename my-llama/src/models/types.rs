use crate::cuda::{CudaBuffer, CudaStream};
use crate::models::compiler::ops::Op;
use crate::utils::parameter::Parameter;
use std::any::Any;

pub trait ModelGraph: Send + Sync {
    fn forward(&mut self, input_tokens: &[u32], stream: &CudaStream) -> Result<CudaBuffer, String>;
    fn as_any(&self) -> &dyn Any;
    fn vocab_size(&self) -> usize;
    fn load_targets(&self, targets: &[i32], stream: &CudaStream);
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView {
    pub offset: usize,
    pub bytes: usize,
    pub batch_size: i32,
    pub out_features: i32,
    pub in_features: i32,
}

pub struct UniversalComputationGraph {
    pub weights: Vec<Parameter>,
    pub activation_arena: CudaBuffer,
    pub pipeline: Vec<Op>,
    pub logits_tensor: TensorView,
    pub targets_offset: usize,
}

impl ModelGraph for UniversalComputationGraph {
    #[inline(always)]
    fn forward(&mut self, input_tokens: &[u32], stream: &CudaStream) -> Result<CudaBuffer, String> {
        let arena_ptr = self.activation_arena.as_raw_ptr();

        let token_bytes = size_of_val(input_tokens);
        let input_gpu_slice = self.activation_arena.slice(0, token_bytes);
        input_gpu_slice.copy_from_host_slice(input_tokens, stream);

        for op in &self.pipeline {
            op.execute(arena_ptr, &self.weights, stream);
        }

        let logits_buffer = self.activation_arena.slice(self.logits_tensor.offset, self.logits_tensor.bytes);
        Ok(logits_buffer)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vocab_size(&self) -> usize {
        self.logits_tensor.out_features as usize
    }

    fn load_targets(&self, targets: &[i32], stream: &CudaStream) {
        let size_in_bytes = size_of_val(targets);
        let targets_gpu_slice = self.activation_arena.slice(self.targets_offset, size_in_bytes);
        targets_gpu_slice.copy_from_host_slice(targets, stream);
    }
}
