use std::ffi::c_void;
use std::ptr;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_rms_norm(output: *mut f32, input: *const f32, weight: *const f32, batch_size: i32, hidden_size: i32, epsilon: f32, s: *mut c_void);
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
}

struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
}

impl CudaBuffer {
    fn alloc(size: usize) -> Self {
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut raw_ptr, size * std::mem::size_of::<f32>());
            assert_eq!(res, 0, "cudaMalloc failed");
        }
        CudaBuffer { ptr: raw_ptr as *mut f32, size }
    }
    fn copy_to_device(&self, host_data: &[f32]) {
        unsafe {
            cudaMemcpy(self.ptr as *mut c_void, host_data.as_ptr() as *const c_void, self.size * 4, 1);
        }
    }
    fn copy_to_host(&self, host_data: &mut [f32]) {
        unsafe {
            cudaMemcpy(host_data.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, self.size * 4, 2);
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr as *mut c_void);
        }
    }
}

fn main() {
    println!("=== БЕНЧМАРК И ВАЛИДАЦИЯ ЯДРА RMSNORM НА RTX 5090 ===");

    let batch_size = 32768;
    let hidden_size = 5120;
    let size = batch_size * hidden_size;

    let total_bytes = (size * 2 + hidden_size) * 4;
    println!("Конфигурация: Batch={}, Hidden={}", batch_size, hidden_size);
    println!("Общий объем обрабатываемых данных: {:.2} МБ", total_bytes as f64 / 1024.0 / 1024.0);

    let epsilon = 1e-6_f32;
    let h_input = vec![0.1f32; size];
    let h_weight = vec![1.2f32; hidden_size];

    let d_out = CudaBuffer::alloc(size);
    let d_in = CudaBuffer::alloc(size);
    let d_weight = CudaBuffer::alloc(hidden_size);

    d_in.copy_to_device(&h_input);
    d_weight.copy_to_device(&h_weight);

    unsafe {
        launch_rms_norm(
            d_out.ptr,
            d_in.ptr,
            d_weight.ptr,
            batch_size as i32,
            hidden_size as i32,
            epsilon,
            ptr::null_mut(),
        );
        cudaDeviceSynchronize();

        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        assert_eq!(cudaEventCreate(&mut start_event), 0);
        assert_eq!(cudaEventCreate(&mut end_event), 0);

        println!("Запуск стресс-теста на 1000 проходов для прогрева шины памяти...");
        cudaEventRecord(start_event, ptr::null_mut());

        let num_iterations = 1000;
        for _ in 0..num_iterations {
            launch_rms_norm(
                d_out.ptr,
                d_in.ptr,
                d_weight.ptr,
                batch_size as i32,
                hidden_size as i32,
                epsilon,
                ptr::null_mut(),
            );
        }

        cudaEventRecord(end_event, ptr::null_mut());
        cudaEventSynchronize(end_event);

        let mut ms = 0.0f32;
        cudaEventElapsedTime(&mut ms, start_event, end_event);

        let avg_ms = ms / num_iterations as f32;
        let seconds = (avg_ms / 1000.0) as f64;

        let bytes_processed = (size * 8) as f64;
        let bandwidth_gbps = (bytes_processed / 1e9) / seconds;

        println!("\n=== РЕЗУЛЬТАТЫ БЕНЧМАРКА ===");
        println!("Общее время серии: {:.2} сек", ms / 1000.0);
        println!("Среднее время выполнения одного слоя: {:.3} мс", avg_ms);
        println!("Эффективная пропускная способность VRAM: {:.2} ГБ/сек", bandwidth_gbps);

        assert_eq!(cudaGetLastError(), 0, "CUDA Error!");
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }

    let mut final_out = vec![0.0f32; size];
    d_out.copy_to_host(&mut final_out);

    let mut sum_sq = 0.0f32;
    for item in h_input.iter().take(hidden_size) {
        sum_sq += item * item;
    }
    let rms_inv = 1.0f32 / (sum_sq / hidden_size as f32 + epsilon).sqrt();
    let expected_0 = h_input[0] * rms_inv * h_weight[0];

    let error = (final_out[0] - expected_0).abs();
    println!("\n--- РЕЗУЛЬТАТЫ МАТЕМАТИЧЕСКОЙ ВАЛИДАЦИИ ---");
    println!("Ожидалось (CPU): {:.6}", expected_0);
    println!("Получено (GPU):  {:.6}", final_out[0]);
    println!("Абсолютная погрешность: {:e}", error);

    if error < 1e-4 {
        println!("🚀 УСПЕХ! Векторизованный RMSNorm прошел тест точности!");
    } else {
        println!("❌ ПРОВАЛ! Обнаружено расхождение слоев.");
    }
}
