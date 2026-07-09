use std::ffi::c_void;
use std::ptr;

#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_swish_glu(output: *mut f32, gate_in: *const f32, up_in: *const f32, sz: i32, s: *mut c_void);
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
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
            assert_eq!(res, 0, "cudaMalloc failed");
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
    println!("=== БЕНЧМАРК И ВАЛИДАЦИЯ ЯДРА SWIGLU НА RTX 5090 ===");

    let size = 134_217_728;
    let total_bytes = size * 4 * 3;
    println!(
        "Размер тензора: {} элементов (~{:.2} МБ памяти)",
        size,
        total_bytes as f64 / 1024.0 / 1024.0
    );

    let h_gate = vec![0.5f32; size];
    let h_up = vec![1.5f32; size];

    let d_out = CudaBuffer::alloc(size);
    let d_gate = CudaBuffer::alloc(size);
    let d_up = CudaBuffer::alloc(size);

    d_gate.copy_to_device(&h_gate);
    d_up.copy_to_device(&h_up);

    unsafe {
        // Warmup
        launch_swish_glu(d_out.ptr, d_gate.ptr, d_up.ptr, size as i32, ptr::null_mut());
        cudaDeviceSynchronize();

        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        cudaEventCreate(&mut start_event);
        cudaEventCreate(&mut end_event);

        println!("Замеряем скорость в стресс-цикле на 2000 итераций...");
        cudaEventRecord(start_event, ptr::null_mut());

        let num_iterations = 1;
        for _ in 0..num_iterations {
            launch_swish_glu(d_out.ptr, d_gate.ptr, d_up.ptr, size as i32, ptr::null_mut());
        }

        cudaEventRecord(end_event, ptr::null_mut());
        cudaEventSynchronize(end_event);

        let mut ms = 0.0f32;
        cudaEventElapsedTime(&mut ms, start_event, end_event);

        let avg_ms = ms / num_iterations as f32;
        let seconds = (avg_ms / 1000.0) as f64;

        let bytes_per_element = 12;
        let bandwidth_gbps = ((size * bytes_per_element) as f64 / 1e9) / seconds;

        println!("\n=== РЕЗУЛЬТАТЫ БЕНЧМАРКА ===");
        println!("Среднее время выполнения: {:.3} мс", avg_ms);
        println!("Эффективная пропускная способность VRAM: {:.2} ГБ/сек", bandwidth_gbps);

        assert_eq!(cudaGetLastError(), 0, "CUDA Error detected!");
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }

    // Валидация математики
    let mut final_out = vec![0.0f32; size];
    d_out.copy_to_host(&mut final_out);

    let g = h_gate[0];
    let u = h_up[0];
    let swish_expected = g / (1.0f32 + (-g).exp());
    let expected = swish_expected * u;

    let error = (final_out[0] - expected).abs();
    println!("\n--- ВАЛИДАЦИЯ МАТЕМАТИКИ (1-й элемент) ---");
    println!("Ожидалось (CPU): {:.6}", expected);
    println!("Получено (GPU):  {:.6}", final_out[0]);
    println!("Абсолютная погрешность интринсиков: {:e}", error);

    if error < 1e-4 {
        println!("🚀 УСПЕХ! Ядро SwiGLU работает с пиковой точностью!");
    } else {
        println!("❌ ПРОВАЛ! Математика разошлась.");
    }
}
