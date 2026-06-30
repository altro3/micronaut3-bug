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
// 1. ТЕСТ: Проверяем асинхронный cuBLAS на Tensor Cores (Row-Major трюк)
// -------------------------------------------------------------------------
#[test]
fn test_cublas_tensor_cores_forward() {
    // Матрица A [2x3], Матрица B [3x2] -> Выход C [2x2]
    const BATCH_SIZE: usize = 2;
    const IN_FEATURES: usize = 3;
    const OUT_FEATURES: usize = 2;

    let stream = CudaStream::new();
    let gpu_a = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);
    let gpu_b = CudaBuffer::new(IN_FEATURES * OUT_FEATURES);
    let gpu_c = CudaBuffer::new(BATCH_SIZE * OUT_FEATURES);

    // Загружаем асимметричные данные, чтобы проверить правильность индексов cuBLAS
    gpu_a.copy_from_host_async(
        &vec![
            1.0, 2.0, 3.0, // Строка 0
            4.0, 5.0, 6.0, // Строка 1
        ],
        &stream,
    );

    gpu_b.copy_from_host_async(
        &vec![
            0.5, 0.1, // Строка 0
            0.2, 0.4, // Строка 1
            0.3, 0.6, // Строка 2
        ],
        &stream,
    );

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

    // Математический расчет на CPU для проверки:
    // Строка 0: [1*0.5 + 2*0.2 + 3*0.3, 1*0.1 + 2*0.4 + 3*0.6] = [1.8, 2.7]
    // Строка 1: [4*0.5 + 5*0.2 + 6*0.3, 4*0.1 + 5*0.4 + 6*0.6] = [4.8, 6.0]
    let expected_c = vec![1.8, 2.7, 4.8, 6.0];

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
// 2. ТЕСТ: Проверяем асинхронную линковку cuBLAS Backward (Градиент весов)
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

    gpu_input.copy_from_host_async(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &stream);

    gpu_d_output.copy_from_host_async(&vec![0.5, -0.2, 0.1, 0.4], &stream);

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

    // Математический расчет dW = X^T * dY на Tensor Cores:
    // Ячейка 0: 1.0 * 0.5 + 4.0 * 0.1 = 0.9
    assert!(
        (host_dw[0] - 0.9).abs() < 1e-4,
        "Ошибка в dW[0]: получили {}",
        host_dw[0]
    );
}

// -------------------------------------------------------------------------
// 3. ТЕСТ: Проверяем асинхронный ArgMax (Поиск лучшего токена)
// -------------------------------------------------------------------------
#[test]
fn test_argmax_kernel_async() {
    const VOCAB_SIZE: usize = 5;
    let stream = CudaStream::new();

    let gpu_logits = CudaBuffer::new(VOCAB_SIZE);

    // Выделяем 1 ячейку i32 во VRAM под результат индекса
    let gpu_output_idx = CudaBuffer::new_int(1);

    // Загружаем логиты, где максимальный элемент (99.9) сидит на индексе 3
    gpu_logits.copy_from_host_async(&vec![-1.2, 0.5, 3.14, 99.9, -0.01], &stream);

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

#[test]
fn test_adamw_optimizer_step_async() {
    const SIZE: usize = 2;
    let stream = CudaStream::new();

    let gpu_w = CudaBuffer::new(SIZE);
    let gpu_g = CudaBuffer::new(SIZE);
    let gpu_m = CudaBuffer::new(SIZE);
    let gpu_v = CudaBuffer::new(SIZE);

    // Инициализируем тестовое состояние весов, градиентов и моментов
    gpu_w.copy_from_host_async(&vec![1.0, 2.0], &stream);
    gpu_g.copy_from_host_async(&vec![0.1, 0.2], &stream);
    gpu_m.copy_from_host_async(&vec![0.0, 0.0], &stream);
    gpu_v.copy_from_host_async(&vec![0.0, 0.0], &stream);

    unsafe {
        launch_adamw(
            gpu_w.as_raw_ptr(),
            gpu_g.as_raw_ptr(),
            gpu_m.as_raw_ptr(),
            gpu_v.as_raw_ptr(),
            SIZE as i32,
            0.1f32,   // lr
            0.9f32,   // beta1
            0.999f32, // beta2
            1e-8f32,  // epsilon
            0.0f32,   // weight_decay
            1.0f32,   // step = 1
            stream.as_raw(),
        );
    }

    let mut updated_weights = vec![0.0f32; SIZE];
    let mut cleared_grads = vec![0.0f32; SIZE];

    gpu_w.copy_to_host_async(&mut updated_weights, &stream);
    gpu_g.copy_to_host_async(&mut cleared_grads, &stream);
    stream.synchronize();

    // Математическая проверка: при m=0, v=0 и step=1:
    // m_new = 0.1 * g, v_new = 0.001 * g^2
    // bias_correction1 = 0.1, bias_correction2 = 0.001
    // m_hat = g, v_hat = g^2 -> m_hat / sqrt(v_hat) = g / g = 1.0
    // w_new = w - lr * 1.0 = w - 0.1
    // Ожидаем: [1.0 - 0.1, 2.0 - 0.1] = [0.9, 1.9]
    assert!(
        (updated_weights[0] - 0.9).abs() < 1e-4,
        "Ошибка в весе 0: получили {}",
        updated_weights[0]
    );
    assert!(
        (updated_weights[1] - 1.9).abs() < 1e-4,
        "Ошибка в весе 1: получили {}",
        updated_weights[1]
    );

    // Проверяем, что встроенное зануление градиентов отработало честно
    assert_eq!(
        cleared_grads,
        vec![0.0f32; SIZE],
        "Оптимизатор не занулил градиенты во VRAM!"
    );
}
