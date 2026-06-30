use my_llama::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_matmul(
        output_matrix: *mut c_void,
        matrix_a: *const c_void,
        matrix_b: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );
    fn launch_matmul_backward_weights(
        d_weights: *mut c_void,
        input: *const c_void,
        d_output: *const c_void,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
        stream: *mut c_void,
    );
    fn launch_argmax(
        output_index: *mut i32,
        logits: *const f32,
        vocab_size: i32,
        stream: *mut c_void,
    );
    fn launch_adamw(
        weights: *mut c_void,
        gradients: *mut c_void,
        m_buf: *mut c_void,
        v_buf: *mut c_void,
        size: i32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        step: f32,
        stream: *mut c_void,
    );
}

// -------------------------------------------------------------------------
// 1. ТЕСТ: Проверяем асинхронный cuBLAS на Tensor Cores
// -------------------------------------------------------------------------
#[test]
fn test_cublas_tensor_cores_forward() {
    const BATCH_SIZE: usize = 2;
    const IN_FEATURES: usize = 3;
    const OUT_FEATURES: usize = 2;

    let stream = CudaStream::new();
    let gpu_a = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);
    let gpu_b = CudaBuffer::new(IN_FEATURES * OUT_FEATURES);
    let gpu_c = CudaBuffer::new(BATCH_SIZE * OUT_FEATURES);

    // ИСПРАВЛЕНИЕ 1: Явно размечаем f32
    gpu_a.copy_from_host_async(&vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &stream);

    gpu_b.copy_from_host_async(&vec![0.5f32, 0.1, 0.2, 0.4, 0.3, 0.6], &stream);

    unsafe {
        launch_matmul(
            gpu_c.as_raw_ptr(),
            gpu_a.as_raw_ptr(),
            gpu_b.as_raw_ptr(),
            BATCH_SIZE as i32,
            OUT_FEATURES as i32,
            IN_FEATURES as i32,
            stream.as_raw(),
        );
    }

    let mut host_c = vec![0.0f32; BATCH_SIZE * OUT_FEATURES];
    gpu_c.copy_to_host_async(&mut host_c, &stream);
    stream.synchronize();

    let expected_c = vec![1.8f32, 2.7, 4.8, 6.0];

    for i in 0..host_c.len() {
        assert!(
            (host_c[i] - expected_c[i]).abs() < 1e-4,
            "cuBLAS выдал неверную матрицу! На индексе {}: ожидали {}, получили {}",
            i,
            expected_c[i],
            host_c[i]
        );
    }
}

// -------------------------------------------------------------------------
// 2. ТЕСТ: Проверяем асинхронную линковку cuBLAS Backward
// -------------------------------------------------------------------------
#[test]
fn test_cublas_backward_weights_math() {
    const BATCH_SIZE: usize = 2;
    const IN_FEATURES: usize = 3;
    const OUT_FEATURES: usize = 2;

    let stream = CudaStream::new();
    let gpu_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);
    let gpu_d_output = CudaBuffer::new(BATCH_SIZE * OUT_FEATURES);
    let gpu_d_weights = CudaBuffer::new(IN_FEATURES * OUT_FEATURES);

    gpu_input.copy_from_host_async(&vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &stream);

    gpu_d_output.copy_from_host_async(&vec![0.5f32, -0.2, 0.1, 0.4], &stream);

    unsafe {
        launch_matmul_backward_weights(
            gpu_d_weights.as_raw_ptr(),
            gpu_input.as_raw_ptr(),
            gpu_d_output.as_raw_ptr(),
            BATCH_SIZE as i32,
            OUT_FEATURES as i32,
            IN_FEATURES as i32,
            stream.as_raw(),
        );
    }

    let mut host_dw = vec![0.0f32; IN_FEATURES * OUT_FEATURES];
    gpu_d_weights.copy_to_host_async(&mut host_dw, &stream);
    stream.synchronize();

    assert!(
        (host_dw[0] - 0.9f32).abs() < 1e-4,
        "Ошибка в dW: получили {}",
        host_dw[0]
    );
}

// -------------------------------------------------------------------------
// 3. ТЕСТ: Проверяем асинхронный ArgMax
// -------------------------------------------------------------------------
#[test]
fn test_argmax_kernel_async() {
    const VOCAB_SIZE: usize = 5;
    let stream = CudaStream::new();

    let gpu_logits = CudaBuffer::new(VOCAB_SIZE);
    let gpu_output_idx = CudaBuffer::new_int(1);

    gpu_logits.copy_from_host_async(&vec![-1.2f32, 0.5, 3.14, 99.9, -0.01], &stream);

    unsafe {
        launch_argmax(
            gpu_output_idx.as_raw_ptr() as *mut i32,
            gpu_logits.as_raw_ptr() as *const f32,
            VOCAB_SIZE as i32,
            stream.as_raw(),
        );
    }

    let mut host_result = vec![0; 1];
    gpu_output_idx.copy_to_host_async(&mut host_result, &stream);
    stream.synchronize();

    assert_eq!(
        host_result[0], 3,
        "ArgMax ошибся! Ожидали индекс токена 3, получили {}",
        host_result[0]
    );
}

// -------------------------------------------------------------------------
// 4. ТЕСТ: Проверяем асинхронный AdamW оптимизатор
// -------------------------------------------------------------------------
#[test]
fn test_adamw_optimizer_step_async() {
    const SIZE: usize = 2;
    let stream = CudaStream::new();

    let gpu_w = CudaBuffer::new(SIZE);
    let gpu_g = CudaBuffer::new(SIZE);
    let gpu_m = CudaBuffer::new(SIZE);
    let gpu_v = CudaBuffer::new(SIZE);

    gpu_w.copy_from_host_async(&vec![1.0f32, 2.0], &stream);
    gpu_g.copy_from_host_async(&vec![0.1f32, 0.2], &stream);
    gpu_m.copy_from_host_async(&vec![0.0f32; SIZE], &stream);
    gpu_v.copy_from_host_async(&vec![0.0f32; SIZE], &stream);

    unsafe {
        launch_adamw(
            gpu_w.as_raw_ptr(),
            gpu_g.as_raw_ptr(),
            gpu_m.as_raw_ptr(),
            gpu_v.as_raw_ptr(),
            SIZE as i32,
            0.1f32,
            0.9f32,
            0.999f32,
            1e-8f32,
            0.0f32,
            1.0f32,
            stream.as_raw(),
        );
    }

    let mut updated_weights = vec![0.0f32; SIZE];
    let mut cleared_grads = vec![0.0f32; SIZE];

    gpu_w.copy_to_host_async(&mut updated_weights, &stream);
    gpu_g.copy_to_host_async(&mut cleared_grads, &stream);
    stream.synchronize();

    assert!(
        (updated_weights[0] - 0.9f32).abs() < 1e-4,
        "Ошибка в весе 0: получили {}",
        updated_weights[0]
    );
    assert!(
        (updated_weights[1] - 1.9f32).abs() < 1e-4,
        "Ошибка в весе 1: получили {}",
        updated_weights[1]
    );
    assert_eq!(
        cleared_grads,
        vec![0.0f32; SIZE],
        "Оптимизатор не занулил градиенты во VRAM!"
    );
}
