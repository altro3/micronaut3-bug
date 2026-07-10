use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    fn init_cublas_infrastructure();
    fn destroy_cublas_infrastructure();
    pub fn launch_matmul(
        output: *mut f32,
        matrix_a: *const f32,
        matrix_b: *const f32,
        batch_size: i32,
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

struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
}

impl CudaBuffer {
    fn alloc(size: usize) -> Self {
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut raw_ptr, size * 4);
            assert_eq!(res, 0);
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
    println!("=== УЛЬТИМАТИВНЫЙ АСИНХРОННЫЙ СТРЕСС-БЕНЧМАРК CUBLAS TF32 НА BLACKWELL ===");

    unsafe {
        init_cublas_infrastructure();
    }

    let batch_size = 512;
    let in_features = 8192;
    let out_features = 27648;

    let size_a = batch_size * in_features;
    let size_b = in_features * out_features;
    let size_c = batch_size * out_features;

    let flops_per_iter = 2.0 * batch_size as f64 * out_features as f64 * in_features as f64;

    println!(
        "Боевая геометрия Qwen-35B: Активации({}x{}) x Веса({}x{})",
        batch_size, in_features, in_features, out_features
    );
    println!(
        "Выделение памяти: ~{:.2} ГБ под тензоры в VRAM",
        ((size_a + size_b + size_c) * 4) as f64 / 1e9
    );

    let h_a = vec![1.0f32; size_a];
    let h_b = vec![0.0002f32; size_b];

    let d_a = CudaBuffer::alloc(size_a);
    let d_b = CudaBuffer::alloc(size_b);
    let d_c = CudaBuffer::alloc(size_c);

    d_a.copy_to_device(&h_a);
    d_b.copy_to_device(&h_b);

    const NUM_ITERATIONS: usize = 200;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        launch_matmul(
            d_c.ptr,
            d_a.ptr,
            d_b.ptr,
            batch_size as i32,
            out_features as i32,
            in_features as i32,
            stream,
        );
        cudaDeviceSynchronize();

        let mut start_events = vec![ptr::null_mut(); NUM_ITERATIONS];
        let mut end_events = vec![ptr::null_mut(); NUM_ITERATIONS];
        for i in 0..NUM_ITERATIONS {
            assert_eq!(cudaEventCreate(&mut start_events[i]), 0);
            assert_eq!(cudaEventCreate(&mut end_events[i]), 0);
        }

        println!("Запуск асинхронной телеметрии... Очередь cuBLAS заполняется.");
        let start_host = Instant::now();

        for i in 0..NUM_ITERATIONS {
            cudaEventRecord(start_events[i], stream);
            launch_matmul(
                d_c.ptr,
                d_a.ptr,
                d_b.ptr,
                batch_size as i32,
                out_features as i32,
                in_features as i32,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time = start_host.elapsed();
        cudaEventSynchronize(*end_events.last().unwrap());
        let total_host_time = start_host.elapsed();

        let mut tflops_v: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
        let mut total_gpu_ms = 0.0_f32;

        for i in 0..NUM_ITERATIONS {
            let mut ms = 0.0_f32;
            cudaEventElapsedTime(&mut ms, start_events[i], end_events[i]);
            total_gpu_ms += ms;

            let seconds = (ms / 1000.0) as f64;
            let tflops = (flops_per_iter / 1e12) / seconds;
            tflops_v.push(tflops);
        }

        tflops_v.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_tf = tflops_v[0];
        let max_tf = tflops_v[NUM_ITERATIONS - 1];
        let median_tf = tflops_v[NUM_ITERATIONS / 2];
        let p95_worst = tflops_v[(NUM_ITERATIONS as f64 * 0.05) as usize];
        let p99_worst = tflops_v[(NUM_ITERATIONS as f64 * 0.01) as usize];
        let avg_tf = (flops_per_iter * NUM_ITERATIONS as f64 / 1e12) / (total_gpu_ms as f64 / 1000.0);

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА TENSOR CORES ===");
        println!("Время отправки очереди (Launch Overhead): {:.6} сек", host_launch_time.as_secs_f32());
        println!("Полное время теста на GPU (по событиям):  {:.2} сек", total_gpu_ms / 1000.0);
        println!("Полное время ожидания хостом (Wall Time):  {:.2} сек", total_host_time.as_secs_f32());
        println!("-------------------------------------------------------");
        println!("🚀 АБСОЛЮТНЫЙ ПИК СКОРОСТИ (Max Performance): {:.2} TFLOPS", max_tf);
        println!("📉 АБСОЛЮТНЫЙ МИНИМУМ (Min Performance):        {:.2} TFLOPS", min_tf);
        println!("-------------------------------------------------------");
        println!("📈 Средняя производительность:                 {:.2} TFLOPS", avg_tf);
        println!("🎯 Медиана (P50 Перцентиль):                {:.2} TFLOPS", median_tf);
        println!("⚠️ Стабильный перформанс (Истинный P95):    {:.2} TFLOPS", p95_worst);
        println!("🚨 Граница просадок (Истинный P99):         {:.2} TFLOPS", p99_worst);
        println!("-------------------------------------------------------");
        println!("Колебания чистой вычислительной мощности:    {:.2} TFLOPS", max_tf - min_tf);

        let mut final_c = vec![0.0f32; size_c];
        d_c.copy_to_host(&mut final_c);

        let expected = 1.0_f64 * 0.0002_f64 * in_features as f64;
        let actual = final_c[0] as f64;
        let error = (actual - expected).abs();

        println!("\n--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ TF32 GEMM ---");
        println!("Ожидалось на CPU (Ground Truth):  {:.7}", expected);
        println!("Получено на GPU (Computed Value): {:.7}", actual);
        println!("Абсолютная погрешность:          {:e}", error);

        if error < 1e-2 {
            println!("\n🚀 ПОБЕДА! cuBLAS TF32 полностью стабилен под боевой нагрузкой слоев!");
        } else {
            println!("\n❌ МАТЕМАТИЧЕСКИЙ ФАКАП: Нарушена точность ядер.");
        }

        assert_eq!(cudaGetLastError(), 0);
        for i in 0..NUM_ITERATIONS {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        cudaStreamDestroy(stream);
        destroy_cublas_infrastructure();
    }
}
