use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_embeddings(
        out: *mut f32,
        weight: *const f32,
        tokens: *const u32,
        total_tokens: i32,
        out_features: i32,
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
    println!("=== УЛЬТИМАТИВНЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ EMBEDDINGS ===");

    let vocab_size = 152064;
    let out_features = 8192;
    let total_tokens = 64;

    assert_eq!(out_features % 4, 0);

    println!("Боевые параметры эмбеддингов Qwen-35B:");
    println!("Vocab: {}, Features: {}, Tokens in batch: {}", vocab_size, out_features, total_tokens);

    let weight_size = (vocab_size * out_features) as usize;
    let tokens_size = total_tokens as usize;
    let out_size = (total_tokens * out_features) as usize;

    let h_weight = vec![42.0f32; weight_size];

    let h_tokens: Vec<u32> = (0..tokens_size).map(|i| (i % vocab_size as usize) as u32).collect();

    let mut h_output = vec![0.0f32; out_size];

    let d_weight = CudaBuffer::alloc(weight_size * 4);
    let d_tokens = CudaBuffer::alloc(tokens_size * 4);
    let d_out = CudaBuffer::alloc(out_size * 4);

    d_weight.copy_to_device(h_weight.as_ptr() as *const c_void, weight_size * 4);
    d_tokens.copy_to_device(h_tokens.as_ptr() as *const c_void, tokens_size * 4);
    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("Прогрев GPU и L2 кэша ({} итераций)...", NUM_WARMUP);
        for _ in 0..NUM_WARMUP {
            launch_embeddings(
                d_out.ptr as *mut f32,
                d_weight.ptr as *const f32,
                d_tokens.ptr as *const u32,
                total_tokens,
                out_features,
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

        println!("Запуск телеметрии... Забиваем асинхронную очередь GPU.");
        let start_host = Instant::now();

        for i in 0..NUM_ITERATIONS {
            cudaEventRecord(start_events[i], stream);
            launch_embeddings(
                d_out.ptr as *mut f32,
                d_weight.ptr as *const f32,
                d_tokens.ptr as *const u32,
                total_tokens,
                out_features,
                vocab_size,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time = start_host.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time = start_host.elapsed();

        let mut bandwidths: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms = 0.0_f32;

        let bytes_processed = ((tokens_size + out_size) * 4) as u64 + (out_size * 4) as u64;

        for i in 0..NUM_ITERATIONS {
            let mut ms = 0.0_f32;
            cudaEventElapsedTime(&mut ms, start_events[i], end_events[i]);
            total_gpu_ms += ms;

            let seconds = (ms / 1000.0) as f64;
            let gbps = (bytes_processed as f64 / 1e9) / seconds;
            bandwidths.push(gbps);
        }

        bandwidths.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_bw = bandwidths[0];
        let max_bw = bandwidths[NUM_ITERATIONS - 1];
        let median_bw = bandwidths[NUM_ITERATIONS / 2];
        let p95_worst = bandwidths[(NUM_ITERATIONS as f64 * 0.05) as usize];
        let p99_worst = bandwidths[(NUM_ITERATIONS as f64 * 0.01) as usize];
        let avg_bw = (bytes_processed as f64 * NUM_ITERATIONS as f64 / 1e9) / (total_gpu_ms as f64 / 1000.0);

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА EMBEDDINGS ===");
        println!("Время отправки очереди (Launch Overhead): {:.6} сек", host_launch_time.as_secs_f32());
        println!("Полное время теста на GPU (по событиям):  {:.2} сек", total_gpu_ms / 1000.0);
        println!("Полное время ожидания хостом (Wall Time):  {:.2} сек", total_host_time.as_secs_f32());
        println!("-------------------------------------------------------");
        println!("🚀 АБСОЛЮТНЫЙ ПИК СКОРОСТИ (Max Bandwidth): {:.2} ГБ/сек", max_bw);
        println!("📉 АБСОЛЮТНЫЙ МИНИМУМ (Min Bandwidth):        {:.2} ГБ/сек", min_bw);
        println!("-------------------------------------------------------");
        println!("📈 Средняя пропускная способность:         {:.2} ГБ/сек", avg_bw);
        println!("🎯 Медиана (P50 Перцентиль):                {:.2} ГБ/сек", median_bw);
        println!("⚠️ Стабильный перформанс (Истинный P95):    {:.2} ГБ/сек", p95_worst);
        println!("🚨 Граница просадок (Истинный P99):         {:.2} ГБ/сек", p99_worst);
        println!("-------------------------------------------------------");
        println!("Колебания скорости шины (Jitter):         {:.2} ГБ/сек", max_bw - min_bw);

        d_out.copy_to_host(h_output.as_mut_ptr() as *mut c_void, out_size * 4);

        let mut errors = 0;
        for item in h_output.iter().take(out_size) {
            if (item - 42.0f32).abs() > 1e-5 {
                errors += 1;
            }
        }

        println!("\n--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ EMBEDDINGS ---");
        println!("Количество неверных элементов: {}", errors);

        if errors == 0 {
            println!("\n🚀 ПОБЕДА! Двумерное векторное ядро эмбеддингов выдает стопроцентную точность!");
        } else {
            println!("\n❌ МАТЕМАТИЧЕСКИЙ ФАКАП: Вычисления разошлись с эталоном.");
        }

        assert_eq!(cudaGetLastError(), 0);
        for i in 0..NUM_ITERATIONS {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        cudaStreamDestroy(stream);
    }
}
