use std::ffi::{c_char, c_void, CStr};
use std::ptr;

#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_adamw(
        w: *mut f32,
        g: *mut f32,
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
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const c_char;

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
            assert_eq!(res, 0, "Ошибка выполнения cudaMalloc. Проверьте инициализацию CUDA.");
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
                1,
            );
        }
    }

    fn copy_to_host(&self, host_data: &mut [f32]) {
        assert!(host_data.len() >= self.size, "Размер хост-буфера меньше буфера GPU");
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
    println!("=== ТЕСТИРОВАНИЕ СКОРОСТИ И ТОЧНОСТИ ЯДРА ADAMW НА RTX 5090 ===");

    let size = 262_553_760;
    let mem_bytes = size * 4 * 4;
    println!(
        "Размер тестового тензора: {} элементов (~{:.2} МБ общая аллокация)",
        size,
        mem_bytes as f64 / 1024.0 / 1024.0
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
            ptr::null_mut(),
        );
        cudaDeviceSynchronize();

        d_w.copy_to_device(&h_w);
        d_g.copy_to_device(&h_g);
        d_m.copy_to_device(&h_m);
        d_v.copy_to_device(&h_v);

        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        assert_eq!(cudaEventCreate(&mut start_event), 0);
        assert_eq!(cudaEventCreate(&mut end_event), 0);

        println!("Запуск боевого бенчмарка ядра в бесконечном цикле на 20 секунд...");
        println!("==> ОТКРЫВАЙ ДИСПЕТЧЕР ЗАДАЧ (Вкладка GPU -> Производительность) <==");

        cudaEventRecord(start_event, ptr::null_mut());

        let num_iterations = 1;
        for _i in 0..num_iterations {
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
                ptr::null_mut(),
            );
        }

        cudaEventRecord(end_event, ptr::null_mut());
        cudaEventSynchronize(end_event);

        let mut milliseconds = 0.0f32;
        cudaEventElapsedTime(&mut milliseconds, start_event, end_event);

        let avg_milliseconds = milliseconds / num_iterations as f32;
        let seconds = (avg_milliseconds / 1000.0) as f64;

        let bytes_processed = size * 32;
        let bandwidth_gbps = (bytes_processed as f64 / 1e9) / seconds;

        println!("\n=== ФИНАЛЬНЫЕ ЗАМЕРЫ ПОСЛЕ ДЛИТЕЛЬНОГО ТЕСТА ===");
        println!("Всего проходов ядра: {}", num_iterations);
        println!("Total время выполнения серии: {:.2} сек", milliseconds / 1000.0);
        println!("Среднее время выполнения одного ядра: {:.3} мс", avg_milliseconds);
        println!("Стабильная пропускная способность VRAM: {:.2} ГБ/сек", bandwidth_gbps);

        let err = cudaGetLastError();
        if err != 0 {
            let c_str = cudaGetErrorString(err);
            let rust_str = CStr::from_ptr(c_str).to_string_lossy();
            println!("\n[КРИТИЧЕСКИЙ СБОЙ CUDA]: Ядро упало с кодом {}: {}", err, rust_str);
            return;
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }

    println!("Выгрузка вычисленных матриц обратно на CPU для валидации...");
    let mut final_w = vec![0.0f32; size];
    let mut final_g = vec![0.0f32; size];
    d_w.copy_to_host(&mut final_w);
    d_g.copy_to_host(&mut final_g);

    let bc1 = 1.0_f32 - beta1.powf(step);
    let bc2 = 1.0_f32 - beta2.powf(step);

    let m_expected = beta1 * h_m[0] + (1.0_f32 - beta1) * h_g[0];
    let v_expected = beta2 * h_v[0] + (1.0_f32 - beta2) * h_g[0] * h_g[0];
    let m_hat = m_expected / bc1;
    let v_hat = v_expected / bc2;
    let w_expected = h_w[0] - lr * ((m_hat / (v_hat.sqrt() + epsilon)) + weight_decay * h_w[0]);

    let error = (final_w[0] - w_expected).abs();

    let gradients_are_zero = final_g.iter().take(1000).all(|&x| x == 0.0f32);

    println!("\n--- РЕЗУЛЬТАТЫ МАТЕМАТИЧЕСКОЙ ВАЛИДАЦИИ ---");
    println!("Ожидалось (CPU Ground Truth): {:.7}", w_expected);
    println!("Получено (GPU Computed Value): {:.7}", final_w[0]);
    println!("Абсолютная ошибка математики: {:e}", error);
    println!("Первые 1000 градиентов успешно занулены: {}", gradients_are_zero);

    if error < 1e-5 && gradients_are_zero {
        println!("\n🚀 УСПЕХ! Нативный FFI-мост AdamW и ядро работают идеально!");
    } else {
        println!("\n❌ ПРОВАЛ! Обнаружено расхождение математики или градиенты не занулены.");
    }
}
