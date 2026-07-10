use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_rope_forward(
        vec: *mut f32,
        positions: *const i32,
        inv_freq: *const f32,
        num_heads: i32,
        head_dim: i32,
        total_tokens: i32,
        stream_ptr: *mut c_void,
    );

    pub fn launch_rope_backward(
        grad_in: *mut f32,
        positions: *const i32,
        inv_freq: *const f32,
        num_heads: i32,
        head_dim: i32,
        total_tokens: i32,
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
            assert_eq!(res, 0, "cudaMalloc failed");
            assert_eq!(raw_ptr as usize % 16, 0, "Alignment failed!");
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
            cudaMemcpy(host_data, self.ptr, bytes, 2);
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
    println!("=== УЛЬТИМАТИВНЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ ROPE (FORWARD + BACKWARD) ===");

    let num_heads = 40;
    let head_dim = 128;
    let total_tokens = 512;

    println!("Боевая геометрия RoPE слоя Qwen-35B:");
    println!("Heads: {}, Head Dim: {}, Tokens in batch: {}", num_heads, head_dim, total_tokens);

    let total_elements = (total_tokens * num_heads * head_dim) as usize;
    let freq_elements = (head_dim / 2) as usize;

    let h_vec = vec![1.0f32; total_elements];
    let mut h_positions = vec![0i32; total_tokens as usize];
    for i in 0..total_tokens as usize {
        h_positions[i] = i as i32;
    }

    let mut h_freq = vec![0.0f32; freq_elements];
    for i in 0..freq_elements {
        h_freq[i] = 1.0f32 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
    }

    let mut h_output_forward = vec![0.0f32; total_elements];
    let mut h_output_backward = vec![0.0f32; total_elements];

    let d_vec = CudaBuffer::alloc(total_elements * 4);
    let d_positions = CudaBuffer::alloc((total_tokens as usize) * 4);
    let d_freq = CudaBuffer::alloc(freq_elements * 4);

    d_vec.copy_to_device(h_vec.as_ptr() as *const c_void, total_elements * 4);
    d_positions.copy_to_device(h_positions.as_ptr() as *const c_void, (total_tokens as usize) * 4);
    d_freq.copy_to_device(h_freq.as_ptr() as *const c_void, freq_elements * 4);

    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("Прогрев GPU и L2 кэша ({} итераций)...", NUM_WARMUP);
        for _ in 0..NUM_WARMUP {
            launch_rope_forward(
                d_vec.ptr as *mut f32,
                d_positions.ptr as *const i32,
                d_freq.ptr as *const f32,
                num_heads,
                head_dim,
                total_tokens,
                stream,
            );
            launch_rope_backward(
                d_vec.ptr as *mut f32,
                d_positions.ptr as *const i32,
                d_freq.ptr as *const f32,
                num_heads,
                head_dim,
                total_tokens,
                stream,
            );
        }
        cudaDeviceSynchronize();
        assert_eq!(cudaGetLastError(), 0);

        let mut start_events = vec![ptr::null_mut(); NUM_ITERATIONS];
        let mut end_events = vec![ptr::null_mut(); NUM_ITERATIONS];
        for i in 0..NUM_ITERATIONS {
            assert_eq!(cudaEventCreate(&mut start_events[i]), 0);
            assert_eq!(cudaEventCreate(&mut end_events[i]), 0);
        }

        println!("Запуск телеметрии FORWARD... Забиваем асинхронную очередь GPU.");
        let start_host = Instant::now();

        for i in 0..NUM_ITERATIONS {
            cudaEventRecord(start_events[i], stream);
            launch_rope_forward(
                d_vec.ptr as *mut f32,
                d_positions.ptr as *const i32,
                d_freq.ptr as *const f32,
                num_heads,
                head_dim,
                total_tokens,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time_fw = start_host.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time_fw = start_host.elapsed();

        let mut bandwidths_fw: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms_fw = 0.0_f32;

        let bytes_processed = (total_elements * 4 * 2) as u64;

        for i in 0..NUM_ITERATIONS {
            let mut ms = 0.0_f32;
            cudaEventElapsedTime(&mut ms, start_events[i], end_events[i]);
            total_gpu_ms_fw += ms;

            let seconds = (ms / 1000.0) as f64;
            let gbps = (bytes_processed as f64 / 1e9) / seconds;
            bandwidths_fw.push(gbps);
        }

        println!("Запуск телеметрии BACKWARD... Забиваем асинхронную очередь GPU.");
        let start_host_bw = Instant::now();

        for i in 0..NUM_ITERATIONS {
            cudaEventRecord(start_events[i], stream);
            launch_rope_backward(
                d_vec.ptr as *mut f32,
                d_positions.ptr as *const i32,
                d_freq.ptr as *const f32,
                num_heads,
                head_dim,
                total_tokens,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time_bw = start_host_bw.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time_bw = start_host_bw.elapsed();

        let mut bandwidths_bw: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms_bw = 0.0_f32;

        for i in 0..NUM_ITERATIONS {
            let mut ms = 0.0_f32;
            cudaEventElapsedTime(&mut ms, start_events[i], end_events[i]);
            total_gpu_ms_bw += ms;

            let seconds = (ms / 1000.0) as f64;
            let gbps = (bytes_processed as f64 / 1e9) / seconds;
            bandwidths_bw.push(gbps);
        }

        bandwidths_fw.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bandwidths_bw.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА ROPE FORWARD ===");
        println!("Время отправки очереди (Launch Overhead): {:.6} сек", host_launch_time_fw.as_secs_f32());
        println!("Полное время на GPU (по событиям):       {:.2} сек", total_gpu_ms_fw / 1000.0);
        println!("Полное время ожидания хостом (Wall Time):  {:.2} сек", total_host_time_fw.as_secs_f32());
        println!("-------------------------------------------------------");
        println!(
            "🚀 АБСОЛЮТНЫЙ ПИК СКОРОСТИ (Max Bandwidth): {:.2} ГБ/сек",
            bandwidths_fw[NUM_ITERATIONS - 1]
        );
        println!(
            "🎯 Медиана (P50 Перцентиль):                {:.2} ГБ/сек",
            bandwidths_fw[NUM_ITERATIONS / 2]
        );
        println!(
            "⚠️ Стабильный перформанс (Истинный P95):    {:.2} ГБ/сек",
            bandwidths_fw[(NUM_ITERATIONS as f64 * 0.05) as usize]
        );

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА ROPE BACKWARD ===");
        println!("Время отправки очереди (Launch Overhead): {:.6} сек", host_launch_time_bw.as_secs_f32());
        println!("Полное время на GPU (по событиям):       {:.2} сек", total_gpu_ms_bw / 1000.0);
        println!("Полное время ожидания хостом (Wall Time):  {:.2} сек", total_host_time_bw.as_secs_f32());
        println!("-------------------------------------------------------");
        println!(
            "🚀 АБСОЛЮТНЫЙ ПИК СКОРОСТИ (Max Bandwidth): {:.2} ГБ/сек",
            bandwidths_bw[NUM_ITERATIONS - 1]
        );
        println!(
            "🎯 Медиана (P50 Перцентиль):                {:.2} ГБ/сек",
            bandwidths_bw[NUM_ITERATIONS / 2]
        );
        println!(
            "⚠️ Стабильный перформанс (Истинный P95):    {:.2} ГБ/сек",
            bandwidths_bw[(NUM_ITERATIONS as f64 * 0.05) as usize]
        );

        println!("\nИзолированный запуск валидации точности...");
        d_vec.copy_to_device(h_vec.as_ptr() as *const c_void, total_elements * 4);

        launch_rope_forward(
            d_vec.ptr as *mut f32,
            d_positions.ptr as *const i32,
            d_freq.ptr as *const f32,
            num_heads,
            head_dim,
            total_tokens,
            stream,
        );
        d_vec.copy_to_host(h_output_forward.as_mut_ptr() as *mut c_void, total_elements * 4);

        launch_rope_backward(
            d_vec.ptr as *mut f32,
            d_positions.ptr as *const i32,
            d_freq.ptr as *const f32,
            num_heads,
            head_dim,
            total_tokens,
            stream,
        );
        d_vec.copy_to_host(h_output_backward.as_mut_ptr() as *mut c_void, total_elements * 4);

        let mut max_diff = 0.0_f32;
        for i in 0..total_elements {
            let diff = (h_output_backward[i] - h_vec[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        println!("\n--- РЕВЕРСИВНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ ROPE (BACKWARD(FORWARD(X)) == X) ---");
        println!("Максимальное тригонометрическое расхождение: {:e}", max_diff);
        if max_diff < 1e-4 {
            println!("\n🚀 ПОБЕДА! Кастомные In-place ядра RoPE полностью обратимы и выдают идеальную точность!");
        } else {
            println!("\n❌ МАТЕМАТИЧЕСКИЙ ФАКАП: Обратный проход рассинхронизировался с прямым.");
        }
        assert_eq!(cudaGetLastError(), 0);
        for i in 0..NUM_ITERATIONS {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        cudaStreamDestroy(stream);
    }
}
