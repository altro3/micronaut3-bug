use crate::cuda::CudaStream;
use crate::models::compiler::ops::sys::launch_adamw;
use crate::models::compiler::ops::Op;
use crate::utils::parameter::Parameter;

pub unsafe fn dispatch_optimizer(op: &Op, weights: &[Parameter], stream: &CudaStream) {
    unsafe {
        match *op {
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
                let w_ptr = weight.data.as_raw_ptr() as *mut f32;
                let g_ptr = weight.grad.as_ref().expect("AdamW требует градиенты").as_raw_ptr() as *mut f32;
                let m_ptr = weight.m_buffer.as_ref().expect("Буфер M не инициализирован").as_raw_ptr() as *mut f32;
                let v_ptr = weight.v_buffer.as_ref().expect("Буфер V не инициализирован").as_raw_ptr() as *mut f32;

                launch_adamw(
                    w_ptr,
                    g_ptr,
                    m_ptr,
                    v_ptr,
                    weight.size as i32,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    step,
                    stream.as_raw(),
                );
            }
            _ => unreachable!(),
        }
    }
}
