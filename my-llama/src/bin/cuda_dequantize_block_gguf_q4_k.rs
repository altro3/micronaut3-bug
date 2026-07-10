#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_dequantize_q4_k(output: *mut f32, input: *const std::ffi::c_void, num_elements: i32, stream_ptr: *mut std::ffi::c_void);
    fn cudaMalloc(dev_ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const std::ffi::c_char;
    fn cudaStreamCreateWithFlags(p_stream: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventRecord(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut std::ffi::c_void, end: *mut std::ffi::c_void) -> i32;
}

#[repr(C, packed)]
struct RustBlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

struct CudaBuffer {
    ptr: *mut std::ffi::c_void,
    size_bytes: usize,
}

impl CudaBuffer {
    fn alloc(size_bytes: usize) -> Self {
        let mut raw_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut raw_ptr, size_bytes);
            assert_eq!(res, 0);
            assert_eq!(raw_ptr as usize % 16, 0);
        }
        CudaBuffer { ptr: raw_ptr, size_bytes }
    }

    fn copy_to_device(&self, host_data: &[u8]) {
        assert!(host_data.len() >= self.size_bytes);
        unsafe {
            cudaMemcpy(self.ptr, host_data.as_ptr() as *const std::ffi::c_void, self.size_bytes, 1);
        }
    }

    fn copy_to_host(&self, host_data: &mut [f32]) {
        assert!(host_data.len() * 4 >= self.size_bytes);
        unsafe {
            cudaMemcpy(host_data.as_mut_ptr() as *mut std::ffi::c_void, self.ptr, self.size_bytes, 2);
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
    println!("=== РАСШИРЕННЫЙ АСИНХРОННЫЙ БЕНЧМАРК ДЕКВАНТОВАНИЯ GGUF Q4_K_M НА BLACKWELL ===");

    let num_blocks = 1_025_600;
    let num_elements = num_blocks * 256;

    let bytes_read = num_blocks as u64 * 144;
    let bytes_written = num_elements as u64 * 4;
    let iter_bytes = bytes_read + bytes_written;

    println!(
        "Конфигурация: {} супер-блоков GGUF (~{:.2} МБ упаковано -> {:.2} МБ FP32 тензор)",
        num_blocks,
        bytes_read as f64 / 1e6,
        bytes_written as f64 / 1e6
    );

    let mut h_input = Vec::with_capacity(num_blocks * size_of::<RustBlockQ4K>());
    for i in 0..num_blocks {
        let mut scales = [0u8; 12];
        scales[0] = 0x2A;
        scales[6] = 0x03;

        let block = RustBlockQ4K { d: 0x3800, dmin: 0x2E00, scales, qs: [0x42; 128] };
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(&block as *const RustBlockQ4K as *const u8, size_of::<RustBlockQ4K>()) };
        h_input.extend_from_slice(bytes);
    }

    let d_input = CudaBuffer::alloc(num_blocks * size_of::<RustBlockQ4K>());
    let d_output = CudaBuffer::alloc(num_elements * size_of::<f32>());

    d_input.copy_to_device(&h_input);

    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        launch_dequantize_q4_k(d_output.ptr as *mut f32, d_input.ptr, num_elements as i32, stream);
        cudaDeviceSynchronize();

        let mut start_events = vec![std::ptr::null_mut(); NUM_ITERATIONS];
        let mut end_events = vec![std::ptr::null_mut(); NUM_ITERATIONS];
        for i in 0..NUM_ITERATIONS {
            assert_eq!(cudaEventCreate(&mut start_events[i]), 0);
            assert_eq!(cudaEventCreate(&mut end_events[i]), 0);
        }

        println!("Запуск телеметрии... Очередь асинхронно заполняется.");
        let start_host = std::time::Instant::now();

        for i in 0..NUM_ITERATIONS {
            cudaEventRecord(start_events[i], stream);
            launch_dequantize_q4_k(d_output.ptr as *mut f32, d_input.ptr, num_elements as i32, stream);
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time = start_host.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time = start_host.elapsed();

        let mut bandwidths: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms = 0.0_f32;

        for i in 0..NUM_ITERATIONS {
            let mut ms = 0.0_f32;
            cudaEventElapsedTime(&mut ms, start_events[i], end_events[i]);
            total_gpu_ms += ms;

            let seconds = (ms / 1000.0) as f64;
            let gbps = (iter_bytes as f64 / 1e9) / seconds;
            bandwidths.push(gbps);
        }

        bandwidths.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_bw = bandwidths[0];
        let max_bw = bandwidths[NUM_ITERATIONS - 1];
        let median_bw = bandwidths[NUM_ITERATIONS / 2];
        let p95_worst = bandwidths[(NUM_ITERATIONS as f64 * 0.05) as usize];
        let p99_worst = bandwidths[(NUM_ITERATIONS as f64 * 0.01) as usize];
        let avg_bw = (iter_bytes as f64 * NUM_ITERATIONS as f64 / 1e9) / (total_gpu_ms as f64 / 1000.0);

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА ДЕКВАНТОВАНИЯ ===");
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
        println!("Колебания скорости (Jitter):              {:.2} ГБ/сек", max_bw - min_bw);

        let mut final_out = vec![0.0f32; num_elements];
        d_output.copy_to_host(&mut final_out);

        let d_val = 0.5_f32;
        let dmin_val = 0.09375_f32;

        let expected_sc = (0x2A & 63) as f32;
        let expected_min_sc = (0x03 & 63) as f32;

        let expected_w1 = d_val * expected_sc * 2.0 - dmin_val * expected_min_sc;
        let expected_w2 = d_val * expected_sc * 4.0 - dmin_val * expected_min_sc;

        let err1 = (final_out[0] - expected_w1).abs();
        let err2 = (final_out[16] - expected_w2).abs();

        println!("\n--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ GGUF Q4_K_M ---");
        println!("Индекс 0 (Младший ниббл) -> Ожидалось: {:.4}, Получено: {:.4}", expected_w1, final_out[0]);
        println!("Индекс 16 (Старший ниббл) -> Ожидалось: {:.4}, Получено: {:.4}", expected_w2, final_out[16]);
        println!("Абсолютная погрешность для младшего ниббла: {:e}", err1);
        println!("Абсолютная погрешность для старшего ниббла: {:e}", err2);

        if err1 < 1e-4 && err2 < 1e-4 {
            println!("\n🚀 ПОБЕДА! Квантование совпало до бита!");
        } else {
            println!("\n❌ МАТЕМАТИЧЕСКИЙ ФАКАП: Вычисления на GPU расходятся с референсом.");
        }

        let err = cudaGetLastError();
        if err != 0 {
            let c_str = cudaGetErrorString(err);
            println!("[CUDA ERROR]: {}", std::ffi::CStr::from_ptr(c_str).to_string_lossy());
        }

        for i in 0..NUM_ITERATIONS {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        cudaStreamDestroy(stream);
    }
}
