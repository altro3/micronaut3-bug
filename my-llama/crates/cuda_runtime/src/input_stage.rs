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
        host_problem_shapes: *const c_void,
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
