use crate::models::attention::cuda_dispatch_attention;
use crate::models::swiglu::cuda_dispatch_swiglu;
use crate::models::TensorView;
use crate::utils::cuda_stream::cudaMemcpyAsync;
use crate::utils::{CudaStream, Parameter};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Embeddings {
        input_id: usize,
        weight_idx: usize,
        output: TensorView,
    },
    RmsNorm {
        input: TensorView,
        weight_idx: usize,
        output: TensorView,
        eps: f32,
    },
    MatMul {
        input: TensorView,
        weight_idx: usize,
        output: TensorView,
    },
    Add {
        a: TensorView,
        b: TensorView,
        out: TensorView,
    },
    SwiGlu {
        input: TensorView,
        output: TensorView,
    },
    RoPEAndAttention {
        input: TensorView,
        layer_idx: usize,
        output: TensorView,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    },
    MambaScan {
        input: TensorView,
        state_weight_idx: usize,
        output: TensorView,
    },
    FusedCrossEntropy {
        logits: TensorView,
        targets_id: usize,
        d_logits: TensorView,
        losses: TensorView,
    },
    MatMulBackward {
        d_output: TensorView,
        input: TensorView,
        weight_idx: usize,
        d_input: TensorView,
    },
    AdamWStep {
        weight_idx: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: f32,
    },
}

unsafe extern "C" {
    fn launch_rms_norm(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        batch_size: i32,
        hidden_size: i32,
        epsilon: f32,
        stream: *mut c_void,
    );

    fn launch_residual(in_out: *mut c_void, res: *const c_void, size: i32, stream: *mut c_void);

    fn launch_fused_cross_entropy(
        logits: *const f32,
        targets: *const i32,
        d_logits: *mut f32,
        losses: *mut f32,
        num_tokens: i32,
        vocab_size: i32,
        stream_ptr: *mut c_void,
    );

    fn launch_matmul_backward_weights(
        d_weights: *mut f32,
        input: *const f32,
        d_output: *const f32,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );

    fn launch_matmul_backward_input(
        d_input: *mut f32,
        d_output: *const f32,
        weights: *const f32,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );
    fn launch_adamw(
        weights: *mut f32,
        gradients: *mut f32,
        m_buffer: *mut f32,
        v_buffer: *mut f32,
        size: i32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        step: f32,
        stream: *mut c_void,
    );

    fn launch_matmul(
        output_matrix: *mut c_void,
        matrix_a: *const c_void,
        matrix_b: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );
}

impl Op {
    #[inline(always)]
    pub unsafe fn execute(
        &self,
        arena_ptr: *mut c_void,
        weights: &[Parameter],
        stream: &CudaStream,
    ) {
        match *self {
            Op::RmsNorm {
                input,
                weight_idx,
                output,
                eps,
            } => {
                let weight = &weights[weight_idx];

                // Все операции с указателями изолируем в unsafe блоке
                let (in_ptr, out_ptr) = unsafe {
                    let inp = (arena_ptr as *const u8).add(input.offset) as *const f32;
                    let out = (arena_ptr as *mut u8).add(output.offset) as *mut f32;
                    (inp, out)
                };

                unsafe {
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
            }

            Op::MatMul {
                input,
                weight_idx,
                output,
            } => {
                let weight = &weights[weight_idx];

                let (in_ptr, out_ptr) = unsafe {
                    let inp = (arena_ptr as *const u8).add(input.offset);
                    let out = (arena_ptr as *mut u8).add(output.offset);
                    (inp, out)
                };

                unsafe {
                    launch_matmul(
                        out_ptr as *mut _,
                        in_ptr as *const _,
                        weight.data.as_raw_ptr(),
                        input.batch_size,
                        input.out_features,
                        input.in_features,
                        stream.as_raw(),
                    );
                }
            }

            Op::Add { a, b, out } => {
                let (a_ptr, b_ptr, out_ptr) = unsafe {
                    let ap = (arena_ptr as *const u8).add(a.offset);
                    let bp = (arena_ptr as *const u8).add(b.offset);
                    let op = (arena_ptr as *mut u8).add(out.offset);
                    (ap, bp, op)
                };
                let total_elements = a.batch_size * a.in_features;

                unsafe {
                    if a.offset != out.offset {
                        let bytes_to_copy = total_elements as usize * 4;
                        cudaMemcpyAsync(
                            out_ptr as *mut _,
                            a_ptr as *const _,
                            bytes_to_copy,
                            2,
                            stream.as_raw(),
                        );
                    }

                    launch_residual(
                        out_ptr as *mut _,
                        b_ptr as *const _,
                        total_elements,
                        stream.as_raw(),
                    );
                }
            }

            Op::Embeddings {
                input_id,
                weight_idx,
                output,
            } => {
                let weight = &weights[weight_idx];
                let row_size_bytes = output.out_features as usize * 4;

                unsafe {
                    let tokens_ptr = (arena_ptr as *const u8).add(input_id) as *const u32;
                    let weight_ptr = weight.data.as_raw_ptr() as *const u8;
                    let out_ptr = (arena_ptr as *mut u8).add(output.offset);

                    let num_tokens = output.batch_size as usize;
                    let tokens_slice = std::slice::from_raw_parts(tokens_ptr, num_tokens);

                    for (i, &token_id) in tokens_slice.iter().enumerate() {
                        let valid_id =
                            if (token_id as usize) < weight.size / (output.out_features as usize) {
                                token_id as usize
                            } else {
                                0
                            };

                        let src_row_ptr = weight_ptr.add(valid_id * row_size_bytes);
                        let dst_row_ptr = out_ptr.add(i * row_size_bytes);

                        cudaMemcpyAsync(
                            dst_row_ptr as *mut _,
                            src_row_ptr as *const _,
                            row_size_bytes,
                            3,
                            stream.as_raw(),
                        );
                    }
                }
            }

            Op::SwiGlu { input, output } => unsafe {
                cuda_dispatch_swiglu(arena_ptr, input, output, stream);
            },

            Op::RoPEAndAttention {
                input,
                layer_idx,
                output,
                num_heads,
                num_kv_heads,
                head_dim,
            } => unsafe {
                cuda_dispatch_attention(
                    arena_ptr,
                    input,
                    layer_idx,
                    output,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    stream,
                );
            },

            Op::MambaScan {
                input,
                state_weight_idx,
                output,
            } => {
                unimplemented!(
                    "Нативное CUDA-ядро Mamba Selective Scan будет интегрировано в будущем"
                );
            }

            Op::FusedCrossEntropy {
                logits,
                targets_id,
                d_logits,
                losses,
            } => {
                let (logits_ptr, d_logits_ptr, losses_ptr) = unsafe {
                    let log = (arena_ptr as *const u8).add(logits.offset) as *const f32;
                    let d_log = (arena_ptr as *mut u8).add(d_logits.offset) as *mut f32;
                    let loss = (arena_ptr as *mut u8).add(losses.offset) as *mut f32;
                    (log, d_log, loss)
                };

                let targets_ptr = unsafe { (arena_ptr as *const u8).add(targets_id) as *const i32 };

                let num_tokens = logits.batch_size;
                let vocab_size = logits.out_features;

                unsafe {
                    launch_fused_cross_entropy(
                        logits_ptr,
                        targets_ptr,
                        d_logits_ptr,
                        losses_ptr,
                        num_tokens,
                        vocab_size,
                        stream.as_raw(),
                    );
                }
            }

            Op::MatMulBackward {
                d_output,
                input,
                weight_idx,
                d_input,
            } => {
                let weight = &weights[weight_idx];

                let (d_out_ptr, in_ptr, d_in_ptr) = unsafe {
                    let d_out = (arena_ptr as *const u8).add(d_output.offset) as *const f32;
                    let inp = (arena_ptr as *const u8).add(input.offset) as *const f32;
                    let d_in = (arena_ptr as *mut u8).add(d_input.offset) as *mut f32;
                    (d_out, inp, d_in)
                };

                let d_weights_ptr = weight
                    .grad
                    .as_ref()
                    .expect("Критическая ошибка: Запрошен backward для слоя без градиентов")
                    .as_raw_ptr() as *mut f32;

                let batch_size = d_output.batch_size;
                let out_features = d_output.out_features;
                let in_features = input.out_features;

                unsafe {
                    launch_matmul_backward_weights(
                        d_weights_ptr,
                        in_ptr,
                        d_out_ptr,
                        batch_size,
                        out_features,
                        in_features,
                        stream.as_raw(),
                    );

                    launch_matmul_backward_input(
                        d_in_ptr,
                        d_out_ptr,
                        weight.data.as_raw_ptr() as *const f32,
                        batch_size,
                        out_features,
                        in_features,
                        stream.as_raw(),
                    );
                }
            }

            Op::AdamWStep {
                weight_idx,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            } => {
                let weight = &weights[weight_idx];

                let (w_ptr, g_ptr, m_ptr, v_ptr) = {
                    let w = weight.data.as_raw_ptr() as *mut f32;

                    let g = weight
                        .grad
                        .as_ref()
                        .expect("Критическая ошибка: AdamW вызван для слоя без градиентов")
                        .as_raw_ptr() as *mut f32;

                    let m = weight
                        .m_buffer
                        .as_ref()
                        .expect("Критическая ошибка: Буфер моментов M не инициализирован")
                        .as_raw_ptr() as *mut f32;

                    let v = weight
                        .v_buffer
                        .as_ref()
                        .expect("Критическая ошибка: Буфер моментов V не инициализирован")
                        .as_raw_ptr() as *mut f32;

                    (w, g, m, v)
                };

                let total_elements = weight.size as i32;

                unsafe {
                    launch_adamw(
                        w_ptr,
                        g_ptr,
                        m_ptr,
                        v_ptr,
                        total_elements,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                        stream.as_raw(),
                    );
                }
            }
        }
    }
}
