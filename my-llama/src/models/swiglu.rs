use crate::models::TensorView;
use crate::utils::CudaStream;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_swish_glu(
        output: *mut f32,
        gate_input: *const f32,
        up_input: *const f32,
        size: i32,
        stream: *mut c_void,
    );
}

pub unsafe fn cuda_dispatch_swiglu(
    arena_ptr: *mut c_void,
    input: TensorView,
    output: TensorView,
    stream: &CudaStream,
) {
    let half_bytes = input.bytes / 2;
    let elements_count = input.batch_size * input.out_features;

    let (gate_ptr, up_ptr, out_ptr) = unsafe {
        let gate = (arena_ptr as *const u8).add(input.offset) as *const f32;
        let up = (arena_ptr as *const u8).add(input.offset + half_bytes) as *const f32;
        let out = (arena_ptr as *mut u8).add(output.offset) as *mut f32;
        (gate, up, out)
    };

    unsafe {
        launch_swish_glu(out_ptr, gate_ptr, up_ptr, elements_count, stream.as_raw());
    }
}
