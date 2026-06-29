use crate::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_matmul(out_c: *mut c_void, mat_a: *const c_void, mat_b: *const c_void, m: i32, n: i32, k: i32);
    fn launch_swish_glu(output: *mut c_void, gate_input: *const c_void, up_input: *const c_void, size: i32);
}

/// Промышленный слой SwiGLU MLP-блока (Архитектура Qwen)
pub struct SwiGlu {
    pub w_gate: CudaBuffer,
    pub w_up: CudaBuffer,
    pub w_down: CudaBuffer,
    in_features: usize,
    hidden_features: usize,
}

impl SwiGlu {
    pub fn new(in_features: usize, hidden_features: usize) -> Self {
        let w_gate = CudaBuffer::new(in_features * hidden_features);
        let w_up = CudaBuffer::new(in_features * hidden_features);
        let w_down = CudaBuffer::new(hidden_features * in_features);

        let init_w = vec![0.5f32; in_features * hidden_features];
        let init_w_down = vec![0.2f32; hidden_features * in_features];

        w_gate.copy_from_host(&init_w);
        w_up.copy_from_host(&init_w);
        w_down.copy_from_host(&init_w_down);

        SwiGlu {
            w_gate,
            w_up,
            w_down,
            in_features,
            hidden_features,
        }
    }

    pub fn forward(&self, output: &CudaBuffer, input: &CudaBuffer, batch_size: usize) {
        let m = batch_size as i32;
        let n = self.hidden_features as i32;
        let k = self.in_features as i32;

        let h_gate = CudaBuffer::new(batch_size * self.hidden_features);
        let h_up = CudaBuffer::new(batch_size * self.hidden_features);
        let h_fused = CudaBuffer::new(batch_size * self.hidden_features);

        unsafe {
            launch_matmul(h_gate.as_raw_ptr(), input.as_raw_ptr(), self.w_gate.as_raw_ptr(), m, n, k);
            launch_matmul(h_up.as_raw_ptr(), input.as_raw_ptr(), self.w_up.as_raw_ptr(), m, n, k);

            let total_fused_elements = (batch_size * self.hidden_features) as i32;
            launch_swish_glu(h_fused.as_raw_ptr(), h_gate.as_raw_ptr(), h_up.as_raw_ptr(), total_fused_elements);

            let n_down = self.in_features as i32;
            let k_down = self.hidden_features as i32;
            launch_matmul(output.as_raw_ptr(), h_fused.as_raw_ptr(), self.w_down.as_raw_ptr(), m, n_down, k_down);
        }
    }
}
