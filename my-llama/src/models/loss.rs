// src/models/loss.rs или src/models/mod.rs
use crate::utils::CudaBuffer;

unsafe extern "C" {
    fn launch_fused_cross_entropy(
        logits: *const f32,
        targets: *const i32,
        d_logits: *mut f32,
        losses: *mut f32,
        num_tokens: i32,
        vocab_size: i32,
    );
}

pub fn calculate_loss(
    logits: &CudaBuffer,
    targets: &CudaBuffer,
    d_logits: &mut CudaBuffer,
    losses: &mut CudaBuffer,
    num_tokens: usize,
    vocab_size: usize,
) {
    assert_eq!(
        logits.len(),
        num_tokens * vocab_size,
        "Неверный размер буфера логитов"
    );
    assert_eq!(
        d_logits.len(),
        num_tokens * vocab_size,
        "Неверный размер буфера градиентов логитов"
    );
    assert_eq!(losses.len(), num_tokens, "Неверный размер буфера лоссов");

    unsafe {
        launch_fused_cross_entropy(
            logits.as_raw_ptr() as *const f32,
            targets.as_raw_ptr() as *const i32,
            d_logits.as_raw_ptr() as *mut f32,
            losses.as_raw_ptr() as *mut f32,
            num_tokens as i32,
            vocab_size as i32,
        );
    }
}
