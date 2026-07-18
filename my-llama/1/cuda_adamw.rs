use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_adamw(
        w: *mut f32,
        g: *const f32,
        m: *mut f32,
        v: *mut f32,
        sz: i32,
        lr: f32,
        b1: f32,
        b2: f32,
        e: f32,
        wd: f32,
        st: f32,
        s: *mut c_void,
    );

    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;

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
            let res = cudaMalloc(&mut raw_ptr, size * size_of::<f32>());
            assert_eq!(res, 0, "Ошибка выполнения cudaMalloc.");
            assert_eq!(raw_ptr as usize % 16, 0, "Критическая ошибка: память не выровнена!");
        }
        CudaBuffer { ptr: raw_ptr as *mut f32, size }
    }

    fn copy_to_device(&self, host_data: &[f32]) {
        unsafe {
            cudaMemcpy(
                self.ptr as *mut c_void,
                host_data.as_ptr() as *const c_void,
                self.size * size_of::<f32>(),
                1,
            );
        }
    }

    fn copy_to_host(&self, host_data: &mut [f32]) {
        unsafe {
            cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.size * size_of::<f32>(),
                2,
            );
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
    println!("=== РАСШИРЕННЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ ADAMW НА BLACKWELL ===");

    let size = 262_553_760;
    let bytes_per_element: u64 = 28;
    let iter_bytes = size as u64 * bytes_per_element;

    let lr = 1e-4_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.95_f32;
    let epsilon = 1e-8_f32;
    let weight_decay = 0.01_f32;
    let step = 12.0_f32;

    let h_w = vec![0.5f32; size];
    let h_g = vec![0.02f32; size];
    let h_m = vec![0.005f32; size];
    let h_v = vec![0.0001f32; size];

    let d_w = CudaBuffer::alloc(size);
    let d_g = CudaBuffer::alloc(size);
    let d_m = CudaBuffer::alloc(size);
    let d_v = CudaBuffer::alloc(size);

    d_w.copy_to_device(&h_w);
    d_g.copy_to_device(&h_g);
    d_m.copy_to_device(&h_m);
    d_v.copy_to_device(&h_v);

    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        launch_adamw(
            d_w.ptr,
            d_g.ptr,
            d_m.ptr,
            d_v.ptr,
            size as i32,
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step,
            stream,
        );
        cudaDeviceSynchronize();

        d_w.copy_to_device(&h_w);
        d_m.copy_to_device(&h_m);
        d_v.copy_to_device(&h_v);

        let mut start_events = vec![ptr::null_mut(); NUM_ITERATIONS];
        let mut end_events = vec![ptr::null_mut(); NUM_ITERATIONS];

        for i in 0..NUM_ITERATIONS {
            assert_eq!(cudaEventCreate(&mut start_events[i]), 0);
            assert_eq!(cudaEventCreate(&mut end_events[i]), 0);
        }

        println!("Запуск телеметрии производительности... Очередь GPU заполняется асинхронно.");
        let start_host = Instant::now();

        for i in 0..NUM_ITERATIONS {
            cudaEventRecord(start_events[i], stream);
            launch_adamw(
                d_w.ptr,
                d_g.ptr,
                d_m.ptr,
                d_v.ptr,
                size as i32,
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                step,
                stream,
            );
            cudaEventRecord(end_events[i], stream);
        }

        let host_launch_time = start_host.elapsed();

        println!("Все ядра отправлены. Ожидание завершения очереди на RTX 5090...");
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

        println!("\n📊 === РЕЗУЛЬТАТЫ ГЛУБОКОГО СТАТИСТИЧЕСКОГО АНАЛИЗА ===");
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

        println!("\nСброс памяти VRAM и запуск одиночного шага для верификации точности...");
        d_w.copy_to_device(&h_w);
        d_m.copy_to_device(&h_m);
        d_v.copy_to_device(&h_v);

        let verification_step = 1.0_f32;

        launch_adamw(
            d_w.ptr,
            d_g.ptr,
            d_m.ptr,
            d_v.ptr,
            size as i32,
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            verification_step,
            stream,
        );
        cudaDeviceSynchronize();

        let mut final_w = vec![0.0f32; size];
        d_w.copy_to_host(&mut final_w);

        let bc1 = 1.0_f32 - beta1.powf(verification_step);
        let bc2 = 1.0_f32 - beta2.powf(verification_step);

        let m_expected = beta1 * h_m[0] + (1.0_f32 - beta1) * h_g[0];
        let v_expected = beta2 * h_v[0] + (1.0_f32 - beta2) * h_g[0] * h_g[0];
        let m_hat = m_expected / bc1;
        let v_hat = v_expected / bc2;
        let w_expected = h_w[0] - lr * ((m_hat / (v_hat.sqrt() + epsilon)) + weight_decay * h_w[0]);

        let error = (final_w[0] - w_expected).abs();

        println!("\n--- РЕЗУЛЬТАТЫ МАТЕМАТИЧЕСКОЙ ВАЛИДАЦИИ ---");
        println!("Ожидалось на CPU (Ground Truth):  {:.7}", w_expected);
        println!("Получено на GPU (Computed Value): {:.7}", final_w[0]);
        println!("Абсолютная погрешность:          {:e}", error);

        if error < 1e-4 {
            println!("\n🚀 ВАЛИДАЦИЯ УСПЕШНА: Ядро вычисляет данные с абсолютной точностью!");
        } else {
            println!("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Математика GPU разошлась с CPU!");
        }

        for i in 0..NUM_ITERATIONS {
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        cudaStreamDestroy(stream);
    }
}
