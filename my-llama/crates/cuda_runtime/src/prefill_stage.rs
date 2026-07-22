use crate::data_types::{DataType, QuantType};
use std::ffi::c_void;

unsafe extern "C" {
    pub fn launch_prefill_linear(
        output_activations: *mut c_void,
        input_activations: *const c_void,
        quantized_weights: *const c_void,
        batch_size_or_tokens: i32,
        hidden_units_out: i32,
        hidden_units_in: i32,
        activation_type: i32,
        quantization_type: i32,
        stream_ptr: *mut c_void,
    );
}

pub unsafe fn prefill_linear(
    output_activations: *mut c_void,
    input_activations: *const c_void,
    quantized_weights: *const c_void,
    batch_size_or_tokens: i32,
    hidden_units_out: i32,
    hidden_units_in: i32,
    activation_type: DataType,
    quantization_type: QuantType,
    stream_ptr: *mut c_void,
) {
    unsafe {
        launch_prefill_linear(
            output_activations,
            input_activations,
            quantized_weights,
            batch_size_or_tokens,
            hidden_units_out,
            hidden_units_in,
            activation_type as i32,
            quantization_type as i32,
            stream_ptr,
        );
    }
}
