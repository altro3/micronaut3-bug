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
    pub hidden_size: usize,
}

impl RmsNorm {
    pub fn new(hidden_size: usize) -> Self {
        let weight = CudaBuffer::new(hidden_size);
        weight.copy_from_host(&vec![1.0f32; hidden_size]);
        RmsNorm {
            weight,
            hidden_size,
        }
    }

    pub fn forward(&self, output: &CudaBuffer, input: &CudaBuffer, batch_size: usize) {
        unsafe {
            launch_rms_norm(
                output.as_raw_ptr(),
                input.as_raw_ptr(),
                self.weight.as_raw_ptr(),
                batch_size as i32,
                self.hidden_size as i32,
                1e-5f32,
            );
        }
    }
}
