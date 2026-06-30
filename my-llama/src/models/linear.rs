use crate::utils::parameter::Parameter;
use crate::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_matmul(
        output_matrix: *mut c_void,
        matrix_a: *const c_void,
        matrix_b: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );
    fn launch_matmul_backward_weights(
        d_weights: *mut c_void,
        input: *const c_void,
        d_output: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );
    fn launch_matmul_backward_input(
        d_input: *mut c_void,
        d_output: *const c_void,
        weights: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );
}

pub struct Linear {
    pub weight: Parameter,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, stream: &CudaStream) -> Self {
        let weight = Parameter::new(in_features * out_features, stream);

        let init_weights = vec![0.5f32; in_features * out_features];
        weight.load_weights_async(&init_weights, stream);

        Linear {
            weight,
            in_features,
            out_features,
        }
    }

    pub fn forward(
        &self,
        output: &CudaBuffer,
        input: &CudaBuffer,
        batch_size: usize,
        stream: &CudaStream,
    ) {
        unsafe {
            launch_matmul(
                output.as_raw_ptr(),
                input.as_raw_ptr(),
                self.weight.data.as_raw_ptr(),
                batch_size as i32,
                self.out_features as i32,
                self.in_features as i32,
                stream.as_raw(),
            );
        }
    }

    pub fn backward(
        &self,
        d_input: &CudaBuffer,
        input: &CudaBuffer,
        d_output: &CudaBuffer,
        batch_size: usize,
        stream: &CudaStream,
    ) {
        let b_size = batch_size as i32;
        let out_f = self.out_features as i32;
        let in_f = self.in_features as i32;

        unsafe {
            launch_matmul_backward_weights(
                self.weight.grad.as_raw_ptr(),
                input.as_raw_ptr(),
                d_output.as_raw_ptr(),
                b_size,
                out_f,
                in_f,
                stream.as_raw(),
            );

            launch_matmul_backward_input(
                d_input.as_raw_ptr(),
                d_output.as_raw_ptr(),
                self.weight.data.as_raw_ptr(),
                b_size,
                out_f,
                in_f,
                stream.as_raw(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_backward_math_async() {
        const BATCH_SIZE: usize = 2;
        const IN_FEATURES: usize = 3;
        const OUT_FEATURES: usize = 2;

        let stream = CudaStream::new();
        let linear_layer = Linear::new(IN_FEATURES, OUT_FEATURES, &stream);

        let gpu_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);
        let gpu_d_output = CudaBuffer::new(BATCH_SIZE * OUT_FEATURES);
        let gpu_d_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);

        gpu_input.copy_from_host_async(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &stream);

        gpu_d_output.copy_from_host_async(&vec![0.5, -0.2, 0.1, 0.4], &stream);

        linear_layer.backward(&gpu_d_input, &gpu_input, &gpu_d_output, BATCH_SIZE, &stream);

        let mut calculated_grads = vec![0.0f32; IN_FEATURES * OUT_FEATURES];
        linear_layer
            .weight
            .grad
            .copy_to_host_async(&mut calculated_grads, &stream);

        stream.synchronize();

        let expected_dw0 = 0.9f32;
        assert!(
            (calculated_grads[0] - expected_dw0).abs() < 1e-5,
            "Математика dW сломалась! Ожидали {}, получили {}",
            expected_dw0,
            calculated_grads[0]
        );
    }
}
