use super::sys::launch_swish_glu;
use crate::cuda::CudaStream;
use crate::models::types::TensorView;
use std::ffi::c_void;

pub fn dispatch_swiglu(arena_ptr: *mut c_void, input: TensorView, output: TensorView, stream: &CudaStream) {
    let half_bytes = input.bytes / 2;
    let elements_count = input.batch_size * input.out_features;

    unsafe {
        let gate_ptr = (arena_ptr as *const u8).add(input.offset) as *const f32;
        let up_ptr = (arena_ptr as *const u8).add(input.offset + half_bytes) as *const f32;
        let out_ptr = (arena_ptr as *mut u8).add(output.offset) as *mut f32;

        launch_swish_glu(out_ptr, gate_ptr, up_ptr, elements_count, stream.as_raw());
    }
}
