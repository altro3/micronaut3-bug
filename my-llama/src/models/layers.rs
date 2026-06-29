use crate::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_rms_norm(
        output: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        batch_size: i32,
        hidden_size: i32,
        epsilon: f32,
    );
}

pub struct RmsNorm {
    pub weight: CudaBuffer,
    pub epsilon: f32,
    pub hidden_size: usize,
}

impl RmsNorm {
    pub fn new(hidden_size: usize) -> Self {
        let weight_buffer = CudaBuffer::new(hidden_size);
        let initial_weights = vec![1.0f32; hidden_size];
        weight_buffer.copy_from_host(&initial_weights);

        RmsNorm {
            weight: weight_buffer,
            epsilon: 1e-5,
            hidden_size,
        }
    }

    pub fn forward(&self, output: &CudaBuffer, input: &CudaBuffer, batch_size: usize) {
        let b_size = batch_size as i32;
        let h_size = self.hidden_size as i32;

        unsafe {
            launch_rms_norm(
                output.as_raw_ptr(),
                input.as_raw_ptr(),
                self.weight.as_raw_ptr(),
                b_size,
                h_size,
                self.epsilon,
            );
        }
    }
}
