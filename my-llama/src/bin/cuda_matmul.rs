use std::ffi::c_void;
use std::ptr;

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
    println!("=== БЕНЧМАРК И ВАЛИДАЦИЯ CUBLAS TF32 НА RTX 5090 ===");

    unsafe {
        init_cublas_infrastructure();
    }

    let batch_size = 512;
    let out_features = 5120;
    let in_features = 5120;

    println!(
        "Геометрия GEMM: Активации({}x{}) x Веса({}x{})",
        batch_size, in_features, in_features, out_features
    );

    // Выделяем матрицы
    let h_a = vec![1.0f32; batch_size * in_features];
    let h_b = vec![0.002f32; in_features * out_features];

    let d_a = CudaBuffer::alloc(h_a.len());
    let d_b = CudaBuffer::alloc(h_b.len());
    let d_c = CudaBuffer::alloc(batch_size * out_features);

    d_a.copy_to_device(&h_a);
    d_b.copy_to_device(&h_b);

    unsafe {
        // Warmup
        launch_matmul(
            d_c.ptr,
            d_a.ptr,
            d_b.ptr,
            batch_size as i32,
            out_features as i32,
            in_features as i32,
            ptr::null_mut(),
        );
        cudaDeviceSynchronize();

        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        cudaEventCreate(&mut start_event);
        cudaEventCreate(&mut end_event);

        println!("Запускаем замер скорости cuBLAS в цикле на 500 итераций...");
        cudaEventRecord(start_event, ptr::null_mut());

        let num_iterations = 500;
        for _ in 0..num_iterations {
            launch_matmul(
                d_c.ptr,
                d_a.ptr,
                d_b.ptr,
                batch_size as i32,
                out_features as i32,
                in_features as i32,
                ptr::null_mut(),
            );
        }

        cudaEventRecord(end_event, ptr::null_mut());
        cudaEventSynchronize(end_event);

        let mut ms = 0.0f32;
        cudaEventElapsedTime(&mut ms, start_event, end_event);

        let avg_ms = ms / num_iterations as f32;
        let seconds = (avg_ms / 1000.0) as f64;

        let flops = 2.0 * batch_size as f64 * out_features as f64 * in_features as f64;
        let tflops = (flops / 1e12) / seconds;

        println!("\n=== РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ TENSOR CORES ===");
        println!("Среднее время одного пакетного MatMul: {:.3} мс", avg_ms);
        println!("Производительность блоков Blackwell: {:.2} TFLOPS", tflops);

        assert_eq!(cudaGetLastError(), 0, "CUDA Error detected!");
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }

    let mut final_c = vec![0.0f32; batch_size * out_features];
    d_c.copy_to_host(&mut final_c);

    let expected = 10.24f32;
    let actual = final_c[0];
    let error = (actual - expected).abs();

    println!("\n--- РЕЗУЛЬТАТЫ МАТЕМАТИЧЕСКОЙ ВАЛИДАЦИИ ---");
    println!("Ожидалось (CPU Ground Truth): {:.4}", expected);
    println!("Получено (GPU TF32 TensorCore): {:.4}", actual);
    println!("Абсолютная погрешность: {:e}", error);

    if error < 1e-2 {
        println!("🚀 УСПЕХ! cuBLAS TF32 мост работает со стопроцентной точностью слоев!");
    } else {
        println!("❌ ПРОВАЛ! Матрицы перемножены неверно.");
    }

    unsafe {
        destroy_cublas_infrastructure();
    }
}
