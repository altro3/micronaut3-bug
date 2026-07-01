use crate::cuda::CudaStream;
use crate::models::compiler::ops::sys::{launch_fused_cross_entropy, launch_matmul_backward_input, launch_matmul_backward_weights};
use crate::models::compiler::ops::Op;
use crate::utils::parameter::Parameter;
use std::ffi::c_void;

pub unsafe fn dispatch_backward(op: &Op, arena_ptr: *mut c_void, weights: &[Parameter], stream: &CudaStream) {
    // КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ПОД RUST 2024: изолируем арифметику указателей и лаунчеры ядер внутри unsafe
    unsafe {
        match *op {
            Op::FusedCrossEntropy {
                logits,
                targets_id,
                d_logits,
                losses,
            } => {
                let logits_ptr = (arena_ptr as *const u8).add(logits.offset) as *const f32;
                let d_logits_ptr = (arena_ptr as *mut u8).add(d_logits.offset) as *mut f32;
                let losses_ptr = (arena_ptr as *mut u8).add(losses.offset) as *mut f32;
                let targets_ptr = (arena_ptr as *const u8).add(targets_id) as *const i32;

                launch_fused_cross_entropy(
                    logits_ptr,
                    targets_ptr,
                    d_logits_ptr,
                    losses_ptr,
                    logits.batch_size,
                    logits.out_features,
                    stream.as_raw(),
                );
            }
            Op::MatMulBackward {
                d_output,
                input,
                weight_idx,
                d_input,
            } => {
                let weight = &weights[weight_idx];
                let d_out_ptr = (arena_ptr as *const u8).add(d_output.offset) as *const f32;
                let in_ptr = (arena_ptr as *const u8).add(input.offset) as *const f32;
                let d_in_ptr = (arena_ptr as *mut u8).add(d_input.offset) as *mut f32;

                let d_weights_ptr = weight.grad.as_ref().expect("Backward вызван для слоя без градиентов").as_raw_ptr() as *mut f32;

                launch_matmul_backward_weights(
                    d_weights_ptr,
                    in_ptr,
                    d_out_ptr,
                    d_output.batch_size,
                    d_output.out_features,
                    input.out_features,
                    stream.as_raw(),
                );
                launch_matmul_backward_input(
                    d_in_ptr,
                    d_out_ptr,
                    weight.data.as_raw_ptr() as *const f32,
                    d_output.batch_size,
                    d_output.out_features,
                    input.out_features,
                    stream.as_raw(),
                );
            }
            _ => unreachable!(),
        }
    }
}
