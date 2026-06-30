pub mod models;
pub mod token;
pub mod utils;

unsafe extern "C" {
    pub fn launch_fused_cross_entropy(
        logits: *const f32,
        targets: *const i32,
        d_logits: *mut f32,
        losses: *mut f32,
        num_tokens: i32,
        vocab_size: i32,
    );
}

pub fn init_framework() {
    println!("[MY-LLAMA] Инициализация высокопроизводительного CUDA-ядра завершена.");
}
