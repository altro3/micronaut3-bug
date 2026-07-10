use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    fn launch_cross_entropy_loss(
        logits: *const f32,
        grads: *mut f32,
        targets: *const i32,
        losses: *mut f32,
        total_tokens: i32,
        vocab_size: i32,
        stream_ptr: *mut c_void,
    );
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaStreamCreateWithFlags(p_stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
}

struct CudaBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
}

impl CudaBuffer {
    fn alloc(size_bytes: usize) -> Self {
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut raw_ptr, size_bytes);
            assert_eq!(res, 0);
            assert_eq!(raw_ptr as usize % 16, 0);
        }
        CudaBuffer { ptr: raw_ptr, size_bytes }
    }
    fn copy_to_device(&self, host_data: *const c_void, bytes: usize) {
        assert!(bytes <= self.size_bytes);
        unsafe {
            cudaMemcpy(self.ptr, host_data, bytes, 1);
        }
    }
    fn copy_to_host(&self, host_data: *mut c_void, bytes: usize) {
        assert!(bytes <= self.size_bytes);
        unsafe {
            cudaMemcpy(host_data, self.ptr as *const c_void, bytes, 2);
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

fn main() {
    println!("=== ЧЕСТНЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ FUSED CROSS ENTROPY ===");

    let vocab_size = 152064;
    let total_tokens = 64;

    assert_eq!(vocab_size % 4, 0);

    println!("Параметры финального слоя Qwen-35B:");
    println!("Vocab Size: {}, Tokens in batch: {}", vocab_size, total_tokens);

    let logits_size = (total_tokens * vocab_size) as usize;
    let targets_size = total_tokens as usize;
    let losses_size = total_tokens as usize;

    let h_logits = vec![1.0f32; logits_size];
    let h_targets: Vec<i32> = (0..targets_size).map(|i| (i % 4) as i32).collect();
    let mut h_losses = vec![0.0f32; losses_size];
    let mut h_output_gradients = vec![0.0f32; logits_size];

    const NUM_BUFFERS: usize = 4;
    let mut d_logits_pool = Vec::with_capacity(NUM_BUFFERS);
    for _ in 0..NUM_BUFFERS {
        let buf = CudaBuffer::alloc(logits_size * 4);
        buf.copy_to_device(h_logits.as_ptr() as *const c_void, logits_size * 4);
        d_logits_pool.push(buf);
    }

    let d_logits_val = CudaBuffer::alloc(logits_size * 4);
    let d_targets = CudaBuffer::alloc(targets_size * 4);
    let d_losses = CudaBuffer::alloc(losses_size * 4);

    d_targets.copy_to_device(h_targets.as_ptr() as *const c_void, targets_size * 4);

    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("Прогрев GPU и L2 кэша ({} итераций)...", NUM_WARMUP);
        for i in 0..NUM_WARMUP {
            let buf_idx = i % NUM_BUFFERS;
            launch_cross_entropy_loss(
                d_logits_pool[buf_idx].ptr as *const f32,
                d_logits_pool[buf_idx].ptr as *mut f32,
                d_targets.ptr as *const i32,
                d_losses.ptr as *mut f32,
                total_tokens,
                vocab_size,
                stream,
            );
        }
        cudaDeviceSynchronize();
        assert_eq!(cudaGetLastError(), 0);

        let mut start_event: *mut c_void = ptr::null_mut();
        let mut end_event: *mut c_void = ptr::null_mut();
        assert_eq!(cudaEventCreate(&mut start_event), 0);
        assert_eq!(cudaEventCreate(&mut end_event), 0);

        println!("Запуск телеметрии через ротацию буферов (Выбиваем L2 кэш)...");
        let start_host = Instant::now();
        cudaEventRecord(start_event, stream);

        for i in 0..NUM_ITERATIONS {
            let buf_idx = i % NUM_BUFFERS;
            launch_cross_entropy_loss(
                d_logits_pool[buf_idx].ptr as *const f32,
                d_logits_pool[buf_idx].ptr as *mut f32,
                d_targets.ptr as *const i32,
                d_losses.ptr as *mut f32,
                total_tokens,
                vocab_size,
                stream,
            );
        }

        cudaEventRecord(end_event, stream);
        let host_launch_time = start_host.elapsed();

        cudaEventSynchronize(end_event);
        let total_host_time = start_host.elapsed();

        let mut total_gpu_ms = 0.0_f32;
        cudaEventElapsedTime(&mut total_gpu_ms, start_event, end_event);

        let single_iter_bytes = (logits_size * 4 * 3 + targets_size * 4 + losses_size * 4) as u64;

        let avg_gpu_seconds = (total_gpu_ms as f64 / 1000.0) / NUM_ITERATIONS as f64;
        let avg_bw = (single_iter_bytes as f64 / 1e9) / avg_gpu_seconds;

        println!("\n📊 === РЕЗУЛЬТАТЫ ЧЕСТНОГО АНАЛИЗА ШИНЫ ПАМЯТИ ===");
        println!("Время отправки очереди (Launch Overhead): {:.6} сек", host_launch_time.as_secs_f32());
        println!(
            "Чистое время выполнения {} итераций на GPU: {:.4} сек",
            NUM_ITERATIONS,
            total_gpu_ms / 1000.0
        );
        println!("Полное время ожидания хостом (Wall Time):  {:.4} сек", total_host_time.as_secs_f32());
        println!("Объем данных на одну итерацию (Честный):  {:.2} МБ", single_iter_bytes as f64 / 1e6);
        println!("-------------------------------------------------------");
        println!("🚀 ИСТИННАЯ СКОРОСТЬ ШИНЫ (Честный Average): {:.2} ГБ/сек", avg_bw);
        println!("-------------------------------------------------------");

        println!("\nИзолированный запуск валидации математической точности...");
        d_logits_val.copy_to_device(h_logits.as_ptr() as *const c_void, logits_size * 4);
        cudaDeviceSynchronize();

        launch_cross_entropy_loss(
            d_logits_val.ptr as *const f32,
            d_logits_val.ptr as *mut f32,
            d_targets.ptr as *const i32,
            d_losses.ptr as *mut f32,
            total_tokens,
            vocab_size,
            stream,
        );
        cudaDeviceSynchronize();

        d_losses.copy_to_host(h_losses.as_mut_ptr() as *mut c_void, losses_size * 4);
        d_logits_val.copy_to_host(h_output_gradients.as_mut_ptr() as *mut c_void, logits_size * 4);

        let expected_loss = (vocab_size as f32).ln();
        let loss_error = h_losses
            .iter()
            .map(|&loss| (loss - expected_loss).abs())
            .fold(0.0_f32, |max_err, diff| max_err.max(diff));

        let expected_gradient_normal = 1.0_f32 / vocab_size as f32;
        let expected_gradient_target = expected_gradient_normal - 1.0_f32;
        let mut grad_errors = 0;

        for (t, &target_val) in h_targets.iter().enumerate() {
            let base = t * vocab_size as usize;
            let target_idx = target_val as usize;
            for v in 0..vocab_size as usize {
                let actual_grad = h_output_gradients[base + v];
                let expected_grad = if v == target_idx { expected_gradient_target } else { expected_gradient_normal };
                if (actual_grad - expected_grad).abs() > 1e-5 {
                    grad_errors += 1;
                }
            }
        }

        println!("\n--- МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ ---");
        println!("Максимальная погрешность функции потерь: {:e}", loss_error);
        println!("Количество неверных элементов градиента: {}", grad_errors);

        if loss_error < 1e-4 && grad_errors == 0 {
            println!("\n🚀 ПОБЕДА! Квантовое ядро считает без математических искажений!");
        } else {
            println!("\n❌ МАТЕМАТИЧЕСКИЙ ФАКАП: Вычисления разошлись с эталоном.");
        }

        assert_eq!(cudaGetLastError(), 0);
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
        cudaStreamDestroy(stream);
    }
}
