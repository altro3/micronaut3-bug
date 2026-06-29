use crate::utils::CudaBuffer;
use crate::utils::Parameter;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_matmul(
        output_matrix: *mut c_void,
        matrix_a: *const c_void,
        matrix_b: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
    );
    fn launch_matmul_backward_weights(
        d_weights: *mut c_void,
        input: *const c_void,
        d_output: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
    );
    fn launch_matmul_backward_input(
        d_input: *mut c_void,
        d_output: *const c_void,
        weights: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
    );
}

pub struct Linear {
    pub weight: Parameter,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Parameter::new(in_features * out_features);

        let init_weights = vec![0.5f32; in_features * out_features];
        weight.data.copy_from_host(&init_weights);

        Linear {
            weight,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, output: &CudaBuffer, input: &CudaBuffer, batch_size: usize) {
        unsafe {
            launch_matmul(
                output.as_raw_ptr(),
                input.as_raw_ptr(),
                self.weight.data.as_raw_ptr(),
                batch_size as i32,
                self.out_features as i32,
                self.in_features as i32,
            );
        }
    }

    pub fn backward(
        &self,
        d_input: &CudaBuffer,
        input: &CudaBuffer,
        d_output: &CudaBuffer,
        batch_size: usize,
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
            );

            launch_matmul_backward_input(
                d_input.as_raw_ptr(),
                d_output.as_raw_ptr(),
                self.weight.data.as_raw_ptr(),
                b_size,
                out_f,
                in_f,
            );
        }
    }
}
