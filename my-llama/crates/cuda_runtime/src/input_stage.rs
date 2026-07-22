use crate::data_types::DataType;
use std::ffi::c_void;

unsafe extern "C" {
    pub fn launch_varlen_embeddings(
        out: *mut c_void,
        weight: *const c_void,
        weight_scales: *const f32,
        tokens: *const u32,
        seq_offsets: *const i32,
        block_table: *const i32,
        slot_mapping: *mut i32,
        max_blocks_per_seq: i32,
        block_size: i32,
        total_tokens: i32,
        out_features: i32,
        vocab_size: i32,
        num_seqs: i32,
        data_type: i32,
        threads_per_block: i32,
        stream: *mut c_void,
    );

    pub fn launch_fused_rmsnorm_forward(
        out: *mut c_void,
        input: *const c_void,
        gamma: *const c_void,
        gamma_scales: *const f32,
        epsilon: f32,
        total_tokens: i32,
        hidden_size: i32,
        data_type: i32,
        threads_per_block: i32,
        stream: *mut c_void,
    );

    pub fn launch_fused_multimodal_projection(
        host_ptr_a: *const *mut c_void,
        host_ptr_b: *const *mut c_void,
        host_ptr_d: *const *mut c_void,
        bias: *const f32,
        weight_scales: *const f32,
        host_problem_shapes: *const i32,
        num_segments: i32,
        vision_hidden_size: i32,
        text_hidden_size: i32,
        tp_rank: i32,
        tp_size: i32,
        data_type: i32,
        workspace_ptr: *mut c_void,
        stream_ptr: *mut c_void,
    );
}

pub unsafe fn varlen_embeddings(
    out: *mut c_void,
    weight: *const c_void,
    weight_scales: *const f32,
    tokens: *const u32,
    seq_offsets: *const i32,
    block_table: *const i32,
    slot_mapping: *mut i32,
    max_blocks_per_seq: i32,
    block_size: i32,
    total_tokens: i32,
    out_features: i32,
    vocab_size: i32,
    num_seqs: i32,
    data_type: DataType,
    threads_per_block: i32,
    stream: *mut c_void,
) {
    unsafe {
        launch_varlen_embeddings(
            out,
            weight,
            weight_scales,
            tokens,
            seq_offsets,
            block_table,
            slot_mapping,
            max_blocks_per_seq,
            block_size,
            total_tokens,
            out_features,
            vocab_size,
            num_seqs,
            data_type as i32,
            threads_per_block,
            stream,
        );
    }
}

pub unsafe fn fused_rmsnorm_forward(
    out: *mut c_void,
    input: *const c_void,
    gamma: *const c_void,
    gamma_scales: *const f32,
    epsilon: f32,
    total_tokens: i32,
    hidden_size: i32,
    data_type: DataType,
    threads_per_block: i32,
    stream: *mut c_void,
) {
    unsafe {
        launch_fused_rmsnorm_forward(
            out,
            input,
            gamma,
            gamma_scales,
            epsilon,
            total_tokens,
            hidden_size,
            data_type as i32,
            threads_per_block,
            stream,
        );
    }
}

pub unsafe fn fused_multimodal_projection(
    host_ptr_a: *const *mut c_void,
    host_ptr_b: *const *mut c_void,
    host_ptr_d: *const *mut c_void,
    bias: *const f32,
    weight_scales: *const f32,
    host_problem_shapes: *const i32,
    num_segments: i32,
    vision_hidden_size: i32,
    text_hidden_size: i32,
    tp_rank: i32,
    tp_size: i32,
    data_type: DataType,
    workspace_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) {
    unsafe {
        launch_fused_multimodal_projection(
            host_ptr_a,
            host_ptr_b,
            host_ptr_d,
            bias,
            weight_scales,
            host_problem_shapes,
            num_segments,
            vision_hidden_size,
            text_hidden_size,
            tp_rank,
            tp_size,
            data_type as i32,
            workspace_ptr,
            stream_ptr,
        );
    }
}
