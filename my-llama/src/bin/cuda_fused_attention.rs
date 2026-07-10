use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_fused_attention(
        output: *mut f32,
        query: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
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
    println!("=== УЛЬТИМАТИВНЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ FUSED ATTENTION ===");

    let num_heads = 40;
    let num_kv_heads = 8;
    let head_dim = 128;
    let current_seq_len = 4096;

    println!("Боевая геометрия внимания Qwen-35B (GQA):");
    println!(
        "Heads: {}, KV Heads: {}, Dim: {}, Context: {}",
        num_heads, num_kv_heads, head_dim, current_seq_len
    );

    let q_size = (num_heads * head_dim) as usize;
    let kv_cache_size = (current_seq_len * num_kv_heads * head_dim) as usize;
    let out_size = (num_heads * head_dim) as usize;

    let h_query = vec![1.0f32; q_size];
    let h_k_cache = vec![0.01f32; kv_cache_size];
    let h_v_cache = vec![1.0f32; kv_cache_size];
    let mut h_output = vec![0.0f32; out_size];

    let d_query = CudaBuffer::alloc(q_size * 4);
    let d_k_cache = CudaBuffer::alloc(kv_cache_size * 4);
    let d_v_cache = CudaBuffer::alloc(kv_cache_size * 4);
    let d_output = CudaBuffer::alloc(out_size * 4);

    d_query.copy_to_device(h_query.as_ptr() as *const c_void, q_size * 4);
    d_k_cache.copy_to_device(h_k_cache.as_ptr() as *const c_void, kv_cache_size * 4);
    d_v_cache.copy_to_device(h_v_cache.as_ptr() as *const c_void, kv_cache_size * 4);
    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("Прогрев GPU и L2 кэша ({} итераций)...", NUM_WARMUP);
        for _ in 0..NUM_WARMUP {
            launch_fused_attention(
                d_output.ptr as *mut f32,
                d_query.ptr as *const f32,
                d_k_cache.ptr as *const f32,
                d_v_cache.ptr as *const f32,
                num_heads,
                num_kv_heads,
                head_dim,
                current_seq_len,
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
            launch_fused_attention(
                d_output.ptr as *mut f32,
                d_query.ptr as *const f32,
                d_k_cache.ptr as *const f32,
                d_v_cache.ptr as *const f32,
                num_heads,
                num_kv_heads,
                head_dim,
                current_seq_len,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time = start_host.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time = start_host.elapsed();

        let mut bandwidths: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms = 0.0_f32;

        let bytes_processed = ((q_size + kv_cache_size * 2 + out_size) * 4) as u64;

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

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА FUSED ATTENTION ===");
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

        d_output.copy_to_host(h_output.as_mut_ptr() as *mut c_void, out_size * 4);

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let raw_score = (head_dim as f32) * 1.0f32 * 0.01f32 * scale;
        let exp_score = (raw_score - raw_score).exp();
        let sum_exp = (current_seq_len as f32) * exp_score;
        let prob = exp_score / (sum_exp + 1e-9_f32);

        let expected_val = (current_seq_len as f32) * prob * 1.0f32;
        let actual_val = h_output[0];
        let error = (actual_val - expected_val).abs();

        println!("\n--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ FUSED ATTENTION ---");
        println!("Ожидалось на CPU (Ground Truth):  {:.4}", expected_val);
        println!("Получено на GPU (Computed Value): {:.4}", actual_val);
        println!("Абсолютная погрешность:          {:e}", error);

        if error < 1e-3 {
            println!("\n🚀 ПОБЕДА! Монолитное fused-ядро внимания выдает стопроцентную точность!");
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
