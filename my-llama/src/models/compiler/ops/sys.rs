use std::ffi::c_void;

unsafe extern "C" {
    // Базовые ядра (Forward)
    pub fn launch_embeddings(out: *mut f32, w: *const f32, t: *const u32, n: i32, h: i32, v: i32, s: *mut c_void);
    pub fn launch_rms_norm(out: *mut f32, inp: *const f32, w: *const f32, b: i32, h: i32, e: f32, s: *mut c_void);
    pub fn launch_matmul(out: *mut c_void, a: *const c_void, b: *const c_void, b_s: i32, o: i32, i: i32, s: *mut c_void);
    pub fn launch_residual(in_out: *mut c_void, res: *const c_void, size: i32, s: *mut c_void);

    // Кастомные нелинейные ядра (SwiGLU & Attention)
    pub fn launch_swish_glu(out: *mut f32, gate: *const f32, up: *const f32, sz: i32, s: *mut c_void);
    pub fn launch_attention_scores(out: *mut c_void, q: *const c_void, k: *const c_void, n_h: i32, n_kv: i32, d_h: i32, len: i32, s: *mut c_void);
    pub fn launch_softmax_attention(scores: *mut c_void, n_h: i32, len: i32, s: *mut c_void);
    pub fn launch_attention_values(out: *mut c_void, probs: *const c_void, v: *const c_void, n_h: i32, n_kv: i32, d_h: i32, len: i32, s: *mut c_void);

    // Ядра обратного прохода (Backward)
    pub fn launch_fused_cross_entropy(l: *const f32, t: *const i32, dl: *mut f32, loss: *mut f32, n: i32, v: i32, s: *mut c_void);
    pub fn launch_matmul_backward_weights(dw: *mut f32, inp: *const f32, dout: *const f32, b: i32, o: i32, i: i32, s: *mut c_void);
    pub fn launch_matmul_backward_input(din: *mut f32, dout: *const f32, w: *const f32, b: i32, o: i32, i: i32, s: *mut c_void);

    // Оптимизатор
    pub fn launch_adamw(
        w: *mut f32,
        g: *mut f32,
        m: *mut f32,
        v: *mut f32,
        sz: i32,
        lr: f32,
        b1: f32,
        b2: f32,
        e: f32,
        wd: f32,
        st: f32,
        s: *mut c_void,
    );
}
