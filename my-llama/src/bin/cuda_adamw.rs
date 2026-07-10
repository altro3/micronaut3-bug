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

    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;

    fn cudaStreamCreateWithFlags(p_stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
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
            assert_eq!(res, 0, "Ошибка выполнения cudaMalloc. Проверьте инициализацию CUDA.");
            assert_eq!(
                raw_ptr as usize % 16,
                0,
                "Критическая ошибка: Драйвер CUDA вернул невыровненный адрес памяти!"
            );
        }
        CudaBuffer { ptr: raw_ptr as *mut f32, size }
    }

    fn copy_to_device(&self, host_data: &[f32]) {
        assert!(host_data.len() >= self.size, "Размер хост-буфера меньше выделенной памяти GPU");
        unsafe {
            cudaMemcpy(
                self.ptr as *mut c_void,
                host_data.as_ptr() as *const c_void,
                self.size * size_of::<f32>(),
                1, // cudaMemcpyHostToDevice
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
    println!("=== ТЕСТИРОВАНИЕ СКОРОСТИ И ТОЧНОСТИ ЯДРА ADAMW НА ARCHITECTURE BLACKWELL ===");

    let size = 262_553_760;
    let total_buffers_size_bytes = size * size_of::<f32>() * 4;
    println!(
        "Размер тестового тензора: {} элементов (~{:.2} ГБ выделено в VRAM под 4 буфера)",
        size,
        total_buffers_size_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );

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

    unsafe {
        // 1. Создаем честный неблокирующий асинхронный CUDA-стрим
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0, "Не удалось создать асинхронный стрим");

        println!("Запуск прогревочного цикла GPU (Warmup)...");
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

        // Сбрасываем память перед бенчмарком
        d_w.copy_to_device(&h_w);
        d_m.copy_to_device(&h_m);
        d_v.copy_to_device(&h_v);

        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        assert_eq!(cudaEventCreate(&mut start_event), 0);
        assert_eq!(cudaEventCreate(&mut end_event), 0);

        let num_iterations = 1000;
        println!("Запуск асинхронного стресс-бенчмарка ядра на {} итераций...", num_iterations);

        let start_host = Instant::now(); // Фиксируем время старта CPU

        // Регистрируем старт в асинхронной очереди GPU
        cudaEventRecord(start_event, stream);

        for _ in 0..num_iterations {
            // Теперь запуски происходят асинхронно. CPU просто пушит команду в очередь видеокарты и идет дальше
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
        }

        // Регистрируем финиш в очереди GPU
        cudaEventRecord(end_event, stream);

        // Вычисляем чистое время, затраченное процессором на отправку 1000 команд
        let duration_host_launch = start_host.elapsed();

        // Ждем, пока GPU физически докрутит все 1000 итераций из очереди
        cudaEventSynchronize(end_event);
        let duration_host_total = start_host.elapsed(); // Полное время с учетом ожидания

        let mut milliseconds = 0.0f32;
        cudaEventElapsedTime(&mut milliseconds, start_event, end_event);

        let avg_milliseconds = milliseconds / num_iterations as f32;
        let seconds_total = (milliseconds / 1000.0) as f64;

        let bytes_per_element: u64 = 28;
        let total_bytes_processed = size as u64 * bytes_per_element * num_iterations as u64;
        let bandwidth_gbps = (total_bytes_processed as f64 / 1e9) / seconds_total;

        println!("\n=== ФИНАЛЬНЫЕ ЗАМЕРЫ ПОСЛЕ АСИНХРОННОГО СТРЕСС-ТЕСТА ===");
        println!(
            "Время, затраченное CPU на отправку всех ядер (Launch Overhead): {:.6} сек",
            duration_host_launch.as_secs_f32()
        );
        println!("Общее время выполнения на GPU (по событиям): {:.2} сек", seconds_total);
        println!(
            "Полное время ожидания хостом (Host Wall Time): {:.2} сек",
            duration_host_total.as_secs_f32()
        );
        println!("Среднее время выполнения одного прохода: {:.3} мс", avg_milliseconds);
        println!("Чистая пропускная способность VRAM: {:.2} ГБ/сек", bandwidth_gbps);

        cudaStreamDestroy(stream);
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }
}
