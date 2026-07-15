use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_fused_sampling(
        token_id: *mut i32,
        logits: *mut f32,
        rand_val: f32,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        vocab_size: i32,
        stream: *mut c_void,
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
    println!("=== УЛЬТИМАТИВНЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ FUSED SAMPLING ===");

    let vocab_size = 152064;
    let temperature = 0.7_f32;
    let top_k = 40;
    let top_p = 0.95_f32;
    let rand_val = 0.15_f32;

    println!("Боевые параметры сэмплинга:");
    println!(
        "Vocab Size: {}, Temp: {}, Top-K: {}, Top-P: {}, Rand Val: {}",
        vocab_size, temperature, top_k, top_p, rand_val
    );

    let mut h_logits = vec![-2.0f32; vocab_size as usize];
    let target_token_idx = 1337;
    h_logits[target_token_idx] = 15.0f32;

    h_logits[42] = 12.0f32;
    h_logits[777] = 10.0f32;

    let mut h_token_id = vec![0_i32; 1];

    let d_logits = CudaBuffer::alloc((vocab_size * 4) as usize);
    let d_token_id = CudaBuffer::alloc(4);

    d_logits.copy_to_device(h_logits.as_ptr() as *const c_void, (vocab_size * 4) as usize);

    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("Прогрев GPU и L2 кэша ({} итераций)...", NUM_WARMUP);
        for _ in 0..NUM_WARMUP {
            launch_fused_sampling(
                d_token_id.ptr as *mut i32,
                d_logits.ptr as *mut f32,
                rand_val,
                temperature,
                top_k,
                top_p,
                vocab_size,
                stream,
            );
        }
        cudaDeviceSynchronize();

        let mut start_events = vec![ptr::null_mut(); NUM_ITERATIONS];
        let mut end_events = vec![ptr::null_mut(); NUM_ITERATIONS];
        for i in 0..NUM_ITERATIONS {
            assert_eq!(cudaEventCreate(&mut start_events[i]), 0);
            assert_eq!(cudaEventCreate(&mut end_events[i]), 0);
        }

        println!("Запуск телеметрии... Вычисляем задержку сэмплинга на Blackwell.");
        let start_host = Instant::now();

        for i in 0..NUM_ITERATIONS {
            d_logits.copy_to_device(h_logits.as_ptr() as *const c_void, (vocab_size * 4) as usize);

            cudaEventRecord(start_events[i], stream);
            launch_fused_sampling(
                d_token_id.ptr as *mut i32,
                d_logits.ptr as *mut f32,
                rand_val,
                temperature,
                top_k,
                top_p,
                vocab_size,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time = start_host.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time = start_host.elapsed();

        let mut latencies_us: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms = 0.0_f32;

        for i in 0..NUM_ITERATIONS {
            let mut ms = 0.0_f32;
            cudaEventElapsedTime(&mut ms, start_events[i], end_events[i]);
            total_gpu_ms += ms;
            latencies_us.push((ms * 1000.0) as f64);
        }

        latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_lat = latencies_us[0];
        let max_lat = latencies_us[NUM_ITERATIONS - 1];
        let median_lat = latencies_us[NUM_ITERATIONS / 2];
        let p95_lat = latencies_us[(NUM_ITERATIONS as f64 * 0.95) as usize];
        let avg_lat: f64 = latencies_us.iter().sum::<f64>() / NUM_ITERATIONS as f64;

        let bytes_processed = ((vocab_size * 4 * 2) + 4) as f64;
        let avg_bandwidth_gbps = (bytes_processed / 1e9) / (avg_lat / 1e6);

        println!("\n📊 === РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА FUSED SAMPLING ===");
        println!("Время отправки очереди хостом (Launch Time): {:.6} сек", host_launch_time.as_secs_f32());
        println!("Полное время теста на GPU (по событиям):     {:.2} сек", total_gpu_ms / 1000.0);
        println!("Полное время ожидания хостом (Wall Time):     {:.2} сек", total_host_time.as_secs_f32());
        println!("Средняя утилизация шины памяти:               {:.2} ГБ/сек", avg_bandwidth_gbps);
        println!("-------------------------------------------------------");
        println!("🚀 МИНИМАЛЬНАЯ ЗАДЕРЖКА (Быстрый проход):   {:.2} us", min_lat);
        println!("📉 МАКСИМАЛЬНАЯ ЗАДЕРЖКА (Хвост очереди):    {:.2} us", max_lat);
        println!("-------------------------------------------------------");
        println!("📈 Среднее время обработки одного токена:  {:.2} us", avg_lat);
        println!("🎯 Медиана (P50 Перцентиль):                {:.2} us", median_lat);
        println!("⚠️ Стабильный перформанс (Истинный P95):    {:.2} us", p95_lat);
        println!("-------------------------------------------------------");

        d_token_id.copy_to_host(h_token_id.as_mut_ptr() as *mut c_void, 4);

        println!("\n--- ВАЛИДАЦИЯ ТОЧНОСТИ ВЫБОРА ТОКЕНА ---");
        println!("Ожидаемый токен-лидер (Ground Truth ID): {}", target_token_idx);
        println!("Выбранный токен на GPU (Sampled Token ID): {}", h_token_id[0]);

        if h_token_id[0] == target_token_idx as i32 {
            println!("\n🎉 ПОБЕДА! Fused-ядро сэмплинга идеально отфильтровало Top-K/Top-P и выбрало верный токен!");
        } else {
            println!("\n❌ МАТЕМАТИЧЕСКИЙ СБОЙ: Сэмплинг выдал неверный ID.");
        }

        assert_eq!(cudaGetLastError(), 0);
        for i in 0..NUM_ITERATIONS {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        cudaStreamDestroy(stream);
    }
}
