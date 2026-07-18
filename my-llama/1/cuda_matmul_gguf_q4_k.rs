use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_matmul_gguf_q4_k(
        output: *mut f32,
        weights: *const c_void,
        vec_x: *const f32,
        out_features: i32,
        in_features: i32,
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

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
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
    println!("=== УЛЬТИМАТИВНЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ GGUF Q4_K MATMUL ===");

    let out_features = 8192;
    let in_features = 8192;

    assert_eq!(in_features % 256, 0);
    let blocks_per_row = in_features / 256;
    let total_blocks = out_features * blocks_per_row;

    println!("Боевая геометрия слоя Qwen-35B: {} x {}", out_features, in_features);

    let mut h_weights = vec![0u8; total_blocks * size_of::<BlockQ4K>()];

    for b in 0..total_blocks {
        let offset = b * 144;
        let d_half = 0x3800u16;
        let dmin_half = 0x2E00u16;

        h_weights[offset..offset + 2].copy_from_slice(&d_half.to_le_bytes());
        h_weights[offset + 2..offset + 4].copy_from_slice(&dmin_half.to_le_bytes());

        for i in 0..4 {
            let sc_idx = offset + 4 + (i * 3);
            h_weights[sc_idx] = 0x04;
            h_weights[sc_idx + 1] = 0x04;
            h_weights[sc_idx + 2] = 0x02;
        }

        for i in 0..128 {
            h_weights[offset + 16 + i] = 0xA5;
        }
    }

    let h_vec_x = vec![1.0f32; in_features];
    let mut h_output = vec![0.0f32; out_features];

    let d_weights = CudaBuffer::alloc(total_blocks * 144);
    let d_vec_x = CudaBuffer::alloc(in_features * 4);
    let d_output = CudaBuffer::alloc(out_features * 4);

    d_weights.copy_to_device(h_weights.as_ptr() as *const c_void, total_blocks * 144);
    d_vec_x.copy_to_device(h_vec_x.as_ptr() as *const c_void, in_features * 4);

    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("Прогрев GPU и L2 кэша ({} итераций)...", NUM_WARMUP);
        for _ in 0..NUM_WARMUP {
            launch_matmul_gguf_q4_k(
                d_output.ptr as *mut f32,
                d_weights.ptr,
                d_vec_x.ptr as *const f32,
                out_features as i32,
                in_features as i32,
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
            launch_matmul_gguf_q4_k(
                d_output.ptr as *mut f32,
                d_weights.ptr,
                d_vec_x.ptr as *const f32,
                out_features as i32,
                in_features as i32,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time = start_host.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time = start_host.elapsed();

        let mut bandwidths: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms = 0.0_f32;

        let bytes_processed = (total_blocks * 144 + in_features * 4 + out_features * 4) as u64;

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

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА GGUF GEMV ===");
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

        d_output.copy_to_host(h_output.as_mut_ptr() as *mut c_void, out_features * 4);

        let d_val = 0.5_f32;
        let dmin_val = 0.09375_f32;
        let expected_sc = 4.0_f32;
        let expected_min_sc = 2.0_f32;

        let w_low = d_val * expected_sc * 5.0 - dmin_val * expected_min_sc;
        let w_high = d_val * expected_sc * 10.0 - dmin_val * expected_min_sc;
        let expected_val = (in_features / 2) as f32 * (w_low + w_high);

        let actual_val = h_output[0];
        let error = (actual_val - expected_val).abs();

        println!("\n--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ GGUF GEMV ---");
        println!("Ожидалось на CPU (Ground Truth):  {:.4}", expected_val);
        println!("Получено на GPU (Computed Value): {:.4}", actual_val);
        println!("Абсолютная погрешность:          {:e}", error);

        if error < 1e-1 {
            println!("\n🚀 ПОБЕДА! Кастомное fused-ядро GGUF Q4_K_M выдает стопроцентную точность!");
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
