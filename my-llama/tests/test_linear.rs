use my_llama::models::linear::Linear;
use my_llama::utils::CudaBuffer;

#[test]
fn test_linear_backward_math() {
    let linear_layer = Linear::new(3, 2);

    let gpu_input = CudaBuffer::new(2 * 3);
    let gpu_d_output = CudaBuffer::new(2 * 2);
    let gpu_d_input = CudaBuffer::new(2 * 3);

    gpu_input.copy_from_host(&vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);

    gpu_d_output.copy_from_host(&vec![
        0.5, -0.2,
        0.1,  0.4,
    ]);

    linear_layer.backward(&gpu_d_input, &gpu_input, &gpu_d_output, 2);

    // 3. Проверка результата
    let calculated_grads = linear_layer.weight.grad.copy_to_host();

    let expected_dw0 = 0.9f32;

    assert!(
        (calculated_grads[0] - expected_dw0).abs() < 1e-5,
        "Математика dW сломалась! Ожидали {}, получили {}", expected_dw0, calculated_grads[0]
    );
}
