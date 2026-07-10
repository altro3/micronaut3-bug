#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_swish_glu(output: *mut f32, gate_input: *const f32, up_input: *const f32, size: i32, stream_ptr: *mut std::ffi::c_void);

    fn cudaMalloc(dev_ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;

    fn cudaStreamCreateWithFlags(p_stream: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventRecord(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut std::ffi::c_void, end: *mut std::ffi::c_void) -> i32;
}

struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
}

impl CudaBuffer {
    fn alloc(size: usize) -> Self {
        let mut raw_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut raw_ptr, size * size_of::<f32>());
            assert_eq!(res, 0, "cudaMalloc failed");
            assert_eq!(raw_ptr as usize % 16, 0, "Memory alignment failed!");
        }
        CudaBuffer { ptr: raw_ptr as *mut f32, size }
    }

    fn copy_to_device(&self, host_data: &[f32]) {
        unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                host_data.as_ptr() as *const std::ffi::c_void,
                self.size * size_of::<f32>(),
                1,
            );
        }
    }

    fn copy_to_host(&self, host_data: &mut [f32]) {
        unsafe {
            cudaMemcpy(
                host_data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                self.size * size_of::<f32>(),
                2,
            );
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr as *mut std::ffi::c_void);
        }
    }
}

fn main() {
    println!("=== РАСШИРЕННЫЙ АСИНХРОННЫЙ БЕНЧМАРК И ВАЛИДАЦИЯ SWIGLU НА BLACKWELL ===");

    let size = 262_553_760;
    let bytes_per_element: u64 = 12;
    let iter_bytes = size as u64 * bytes_per_element;

    let h_gate = vec![1.2f32; size];
    let h_up = vec![0.5f32; size];

    let d_out = CudaBuffer::alloc(size);
    let d_gate = CudaBuffer::alloc(size);
    let d_up = CudaBuffer::alloc(size);

    d_gate.copy_to_device(&h_gate);
    d_up.copy_to_device(&h_up);

    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        launch_swish_glu(d_out.ptr, d_gate.ptr, d_up.ptr, size as i32, stream);
        cudaDeviceSynchronize();

        let mut start_events = vec![std::ptr::null_mut(); NUM_ITERATIONS];
        let mut end_events = vec![std::ptr::null_mut(); NUM_ITERATIONS];
        for i in 0..NUM_ITERATIONS {
            assert_eq!(cudaEventCreate(&mut start_events[i]), 0);
            assert_eq!(cudaEventCreate(&mut end_events[i]), 0);
        }

        println!("Запуск телеметрии SwiGLU... Забиваем очередь GPU.");
        let start_host = std::time::Instant::now();

        for i in 0..NUM_ITERATIONS {
            cudaEventRecord(start_events[i], stream);
            launch_swish_glu(d_out.ptr, d_gate.ptr, d_up.ptr, size as i32, stream);
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

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА SWIGLU ===");
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

        let mut final_out = vec![0.0f32; size];
        d_out.copy_to_host(&mut final_out);

        let g_val = h_gate[0];
        let u_val = h_up[0];
        let expected_swish = g_val * (1.0_f32 / (1.0_f32 + (-g_val).exp()));
        let expected_out = expected_swish * u_val;

        let error = (final_out[0] - expected_out).abs();
        println!("\n--- РЕЗУЛЬТАТЫ МАТЕМАТИЧЕСКОЙ ВАЛИДАЦИИ SWIGLU ---");
        println!("Ожидалось на CPU (Ground Truth):  {:.7}", expected_out);
        println!("Получено на GPU (Computed Value): {:.7}", final_out[0]);
        println!("Абсолютная погрешность:          {:e}", error);

        if error < 1e-4 {
            println!("\n🚀 ВАЛИДАЦИЯ УСПЕШНА: SwiGLU считает идеально!");
        } else {
            println!("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Математика активации разошлась!");
        }

        for i in 0..NUM_ITERATIONS {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        cudaStreamDestroy(stream);
    }
}
