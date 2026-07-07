use super::sys::{launch_attention_scores, launch_attention_values, launch_softmax_attention};
use crate::cuda::CudaStream;
use crate::models::types::TensorView;
use std::ffi::c_void;

pub fn dispatch_attention(
    arena_ptr: *mut c_void,
    input: TensorView,
    output: TensorView,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: &CudaStream,
) {
    unsafe {
        let in_ptr = (arena_ptr as *const u8).add(input.offset) as *const c_void;
        let out_ptr = (arena_ptr as *mut u8).add(output.offset) as *mut c_void;
        let current_seq_len = input.batch_size;

        let scores_tmp_ptr = out_ptr;

        launch_attention_scores(
            scores_tmp_ptr,
            in_ptr,
            in_ptr,
            num_heads,
            num_kv_heads,
            head_dim,
            current_seq_len,
            stream.as_raw(),
        );

        launch_softmax_attention(scores_tmp_ptr, num_heads, current_seq_len, stream.as_raw());

        launch_attention_values(
            out_ptr,
            scores_tmp_ptr,
            in_ptr,
            num_heads,
            num_kv_heads,
            head_dim,
            current_seq_len,
            stream.as_raw(),
        );
    }
}
