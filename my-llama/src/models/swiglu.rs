use crate::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_matmul(
        out_c: *mut c_void,
        mat_a: *const c_void,
        mat_b: *const c_void,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    );
    fn launch_swish_glu(
        output: *mut c_void,
        gate_input: *const c_void,
        up_input: *const c_void,
        size: i32,
        stream: *mut c_void,
    );
}

pub struct SwiGlu {
    pub w_gate: CudaBuffer,
    pub w_up: CudaBuffer,
    pub w_down: CudaBuffer,
    in_features: usize,
    hidden_features: usize,
}

impl SwiGlu {
    pub fn new(in_features: usize, hidden_features: usize, stream: &CudaStream) -> Self {
        let w_gate = CudaBuffer::new(in_features * hidden_features);
        let w_up = CudaBuffer::new(in_features * hidden_features);
        let w_down = CudaBuffer::new(hidden_features * in_features);

        let init_w = vec![0.5f32; in_features * hidden_features];
        let init_w_down = vec![0.2f32; hidden_features * in_features];

        w_gate.copy_from_host_async(&init_w, stream);
        w_up.copy_from_host_async(&init_w, stream);
        w_down.copy_from_host_async(&init_w_down, stream);

        SwiGlu {
            w_gate,
            w_up,
            w_down,
            in_features,
            hidden_features,
        }
    }

    pub fn forward(
        &self,
        output: &CudaBuffer,
        input: &CudaBuffer,
        batch_size: usize,
        workspace: &CudaBuffer,
        stream: &CudaStream,
    ) {
        let m = batch_size as i32;
        let n = self.hidden_features as i32;
        let k = self.in_features as i32;
        let size = batch_size * self.hidden_features;

        let h_gate = workspace.slice(0, size);
        let h_up = workspace.slice(size, size);
        let h_fused = workspace.slice(size * 2, size);

        unsafe {
            launch_matmul(
                h_gate.as_raw_ptr(),
                input.as_raw_ptr(),
                self.w_gate.as_raw_ptr(),
                m,
                n,
                k,
                stream.as_raw(),
            );
            launch_matmul(
                h_up.as_raw_ptr(),
                input.as_raw_ptr(),
                self.w_up.as_raw_ptr(),
                m,
                n,
                k,
                stream.as_raw(),
            );

            let total_fused_elements = size as i32;
            launch_swish_glu(
                h_fused.as_raw_ptr(),
                h_gate.as_raw_ptr(),
                h_up.as_raw_ptr(),
                total_fused_elements,
                stream.as_raw(),
            );

            let n_down = self.in_features as i32;
            let k_down = self.hidden_features as i32;
            launch_matmul(
                output.as_raw_ptr(),
                h_fused.as_raw_ptr(),
                self.w_down.as_raw_ptr(),
                m,
                n_down,
                k_down,
                stream.as_raw(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::CudaStream;

    #[test]
    fn test_swiglu_forward_math_async() {
        const BATCH_SIZE: usize = 1;
        const IN_FEATURES: usize = 2;
        const HIDDEN_FEATURES: usize = 2;

        let stream = CudaStream::new();
        let swiglu = SwiGlu::new(IN_FEATURES, HIDDEN_FEATURES, &stream);

        let gpu_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);
        let gpu_output = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);

        let gpu_workspace = CudaBuffer::new(3 * BATCH_SIZE * HIDDEN_FEATURES);

        gpu_input.copy_from_host_async(&vec![1.0f32, 2.0f32], &stream);

        swiglu.forward(&gpu_output, &gpu_input, BATCH_SIZE, &gpu_workspace, &stream);

        let mut host_output = vec![0.0f32; BATCH_SIZE * IN_FEATURES];
        gpu_output.copy_to_host_async(&mut host_output, &stream);

        stream.synchronize();

        println!("[ОТЛАДКА SWIGLU] Выход слоя с GPU: {:?}", host_output);
        let expected_val = 0.735817f32;

        assert!(
            (host_output[0] - expected_val).abs() < 1e-4,
            "Математика SwiGLU нарушена! Ожидали {}, получили {}",
            expected_val,
            host_output[0]
        );
        assert!(
            (host_output[1] - expected_val).abs() < 1e-4,
            "Математика SwiGLU нарушена во второй ячейке!"
        );

        println!("[ЮНИТ-ТЕСТ УСПЕШЕН] Асинхронный SwiGLU на статической памяти работает идеально.");
    }
}
