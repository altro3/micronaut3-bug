use crate::utils::{CudaBuffer, CudaStream};

unsafe extern "C" {
    fn launch_fused_cross_entropy(
        logits: *const f32,
        targets: *const i32,
        d_logits: *mut f32,
        losses: *mut f32,
        num_tokens: i32,
        vocab_size: i32,
        stream: *mut std::ffi::c_void,
    );
}

pub fn calculate_loss(
    logits: &CudaBuffer,
    targets: &CudaBuffer,
    d_logits: &mut CudaBuffer,
    losses: &mut CudaBuffer,
    num_tokens: usize,
    vocab_size: usize,
    stream: &CudaStream,
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
            stream.as_raw(),
        );
    }
}

// src/models/loss.rs

// =========================================================================
// ИЗОЛИРОВАННЫЙ АСИНХРОННЫЙ ЮНИТ-ТЕСТ МАТЕМАТИКИ LOSS ФУНКЦИИ
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_cross_entropy_math_async() {
        const NUM_TOKENS: usize = 2;
        const VOCAB_SIZE: usize = 2;

        let stream = CudaStream::new();

        let gpu_logits = CudaBuffer::new(NUM_TOKENS * VOCAB_SIZE);
        let gpu_targets = CudaBuffer::new_int(NUM_TOKENS);
        let mut gpu_d_logits = CudaBuffer::new(NUM_TOKENS * VOCAB_SIZE);
        let mut gpu_losses = CudaBuffer::new(NUM_TOKENS);

        // ИСПРАВЛЕНИЕ 1: Явно размечаем тип f32 для логитов модели
        gpu_logits.copy_from_host_async(&vec![1.5f32, 1.5, 1.5, 1.5], &stream);
        gpu_targets.copy_from_host_async(&vec![0, 1], &stream);

        // Запускаем расчет лосса в асинхронном стриме
        calculate_loss(
            &gpu_logits,
            &gpu_targets,
            &mut gpu_d_logits,
            &mut gpu_losses,
            NUM_TOKENS,
            VOCAB_SIZE,
            &stream,
        );

        let mut host_losses = vec![0.0f32; NUM_TOKENS];
        let mut host_d_logits = vec![0.0f32; NUM_TOKENS * VOCAB_SIZE];

        gpu_losses.copy_to_host_async(&mut host_losses, &stream);
        gpu_d_logits.copy_to_host_async(&mut host_d_logits, &stream);

        // Точка синхронизации перед проверкой ассертов
        stream.synchronize();

        let expected_loss = std::f32::consts::LN_2;
        assert!(
            (host_losses[0] - expected_loss).abs() < 1e-5,
            "Ошибка в расчете лосса Токена 0"
        );
        assert!(
            (host_losses[1] - expected_loss).abs() < 1e-5,
            "Ошибка в расчете лосса Токена 1"
        );

        // ИСПРАВЛЕНИЕ 2: Проверяем градиенты строго через дельту допуска .abs() < 1e-5
        let expected_gradients = vec![-0.5f32, 0.5, 0.5, -0.5];
        for i in 0..(NUM_TOKENS * VOCAB_SIZE) {
            assert!(
                (host_d_logits[i] - expected_gradients[i]).abs() < 1e-5,
                "Искажение градиента лосса на индексе {}: ожидали {}, получили {}",
                i,
                expected_gradients[i],
                host_d_logits[i]
            );
        }

        println!(
            "[ЮНИТ-ТЕСТ УСПЕШЕН] Fused Cross-Entropy Loss полностью подтвердил математическую точность."
        );
    }
}
