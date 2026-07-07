use super::interface::Op;
use super::sys::{launch_embeddings, launch_matmul, launch_residual, launch_rms_norm};
use crate::cuda::sys::CUDA_MEMCPY_DEVICE_TO_DEVICE;
use crate::cuda::{CudaStream, sys::cudaMemcpyAsync};
use crate::utils::parameter::Parameter;
use std::ffi::c_void;

pub fn dispatch_forward(op: &Op, arena_ptr: *mut c_void, weights: &[Parameter], stream: &CudaStream) {
    match *op {
        Op::Embeddings { input_id, weight_idx, output } => {
            let weight = &weights[weight_idx];

            unsafe {
                let tokens_ptr = (arena_ptr as *const u8).add(input_id) as *const u32;
                let out_ptr = (arena_ptr as *mut u8).add(output.offset) as *mut f32;

                launch_embeddings(
                    out_ptr,
                    weight.data.as_raw_ptr() as *const f32,
                    tokens_ptr,
                    output.batch_size,
                    output.out_features,
                    (weight.size / output.out_features as usize) as i32,
                    stream.as_raw(),
                );
            }
        },
        Op::RmsNorm { input, weight_idx, output, eps } => {
            let weight = &weights[weight_idx];

            unsafe {
                let in_ptr = (arena_ptr as *const u8).add(input.offset) as *const f32;
                let out_ptr = (arena_ptr as *mut u8).add(output.offset) as *mut f32;

                launch_rms_norm(
                    out_ptr,
                    in_ptr,
                    weight.data.as_raw_ptr() as *const f32,
                    input.batch_size,
                    input.out_features,
                    eps,
                    stream.as_raw(),
                );
            }
        },
        Op::MatMul { input, weight_idx, output } => {
            let weight = &weights[weight_idx];

            unsafe {
                let in_ptr = (arena_ptr as *const u8).add(input.offset) as *const c_void;
                let out_ptr = (arena_ptr as *mut u8).add(output.offset) as *mut c_void;

                launch_matmul(
                    out_ptr,
                    in_ptr,
                    weight.data.as_raw_ptr(),
                    input.batch_size,
                    input.out_features,
                    input.in_features,
                    stream.as_raw(),
                );
            }
        },
        Op::Add { a, b, out } => {
            let total_elements = a.batch_size * a.in_features;

            unsafe {
                let a_ptr = (arena_ptr as *const u8).add(a.offset) as *const c_void;
                let b_ptr = (arena_ptr as *const u8).add(b.offset) as *const c_void;
                let out_ptr = (arena_ptr as *mut u8).add(out.offset) as *mut c_void;

                if a.offset != out.offset {
                    cudaMemcpyAsync(out_ptr, a_ptr, total_elements as usize * 4, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream.as_raw());
                }
                launch_residual(out_ptr, b_ptr, total_elements, stream.as_raw());
            }
        },
        _ => unreachable!(),
    }
}
