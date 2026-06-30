use crate::models::TensorView;
use crate::utils::CudaStream;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_attention_scores(
        output_scores: *mut c_void,
        query: *const c_void,
        k_cache: *const c_void,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
        stream: *mut c_void,
    );
    fn launch_softmax_attention(
        scores: *mut c_void,
        num_heads: i32,
        current_seq_len: i32,
        stream: *mut c_void,
    );
    fn launch_attention_values(
        output: *mut c_void,
        probabilities: *const c_void,
        v_cache: *const c_void,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
        stream: *mut c_void,
    );
}

pub unsafe fn cuda_dispatch_attention(
    arena_ptr: *mut c_void,
    input: TensorView,
    _layer_idx: usize,
    output: TensorView,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: &CudaStream,
) {
    let (in_ptr, out_ptr) = unsafe {
        let inp = (arena_ptr as *const u8).add(input.offset) as *const c_void;
        let out = (arena_ptr as *mut u8).add(output.offset) as *mut c_void;
        (inp, out)
    };

    let scores_tmp_ptr = out_ptr;
    let current_seq_len = input.batch_size;

    unsafe {
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
