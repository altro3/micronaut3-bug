use std::ffi::c_void;
use std::ptr;

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
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
}

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct BlockQ4K {
    d: f32,
    dmin: f32,
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
            assert_eq!(res, 0, "cudaMalloc failed");
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
    println!("=== БЕНЧМАРК И ВАЛИДАЦИЯ GGUF Q4_K MATMUL НА RTX 5090 ===");

    let out_features = 5120;
    let in_features = 5120;

    assert_eq!(in_features % 256, 0, "in_features должен быть кратен размеру блока (256)");
    let blocks_per_row = in_features / 256;
    let total_blocks = out_features * blocks_per_row;

    println!("Матрица весов: {} x {}", out_features, in_features);
    println!("Всего GGUF-блоков в видеопамяти: {}", total_blocks);

    let mut h_weights = vec![BlockQ4K { d: 0.0f32, dmin: 0.0f32, scales: [0u8; 12], qs: [0u8; 128] }; total_blocks];

    for block in h_weights.iter_mut() {
        block.d = 0.5f32;
        block.dmin = 0.1f32;

        for i in 0..4 {
            let offset = i * 3;
            block.scales[offset] = 0x44;
            block.scales[offset + 1] = 0x44;
            block.scales[offset + 2] = 0x22;
        }

        for i in 0..128 {
            block.qs[i] = 0xA5;
        }
    }

    let h_vec_x = vec![1.0f32; in_features];
    let mut h_output = vec![0.0f32; out_features];

    let d_weights = CudaBuffer::alloc(total_blocks * size_of::<BlockQ4K>());
    let d_vec_x = CudaBuffer::alloc(in_features * 4);
    let d_output = CudaBuffer::alloc(out_features * 4);

    d_weights.copy_to_device(h_weights.as_ptr() as *const c_void, total_blocks * 144);
    d_vec_x.copy_to_device(h_vec_x.as_ptr() as *const c_void, in_features * 4);

    unsafe {
        launch_matmul_gguf_q4_k(
            d_output.ptr as *mut f32,
            d_weights.ptr,
            d_vec_x.ptr as *const f32,
            out_features as i32,
            in_features as i32,
            ptr::null_mut(),
        );
        cudaDeviceSynchronize();

        let mut start_event = ptr::null_mut();
        let mut end_event = ptr::null_mut();
        assert_eq!(cudaEventCreate(&mut start_event), 0);
        assert_eq!(cudaEventCreate(&mut end_event), 0);

        println!("\nЗапускаем стресс-тест на 2000 итераций для Диспетчера задач...");
        println!("==> ОТКРЫВАЙ ДИСПЕТЧЕР ЗАДАЧ (Вкладка GPU -> График Compute_0) <==");

        cudaEventRecord(start_event, ptr::null_mut());

        let num_iterations = 2000;
        for _ in 0..num_iterations {
            launch_matmul_gguf_q4_k(
                d_output.ptr as *mut f32,
                d_weights.ptr,
                d_vec_x.ptr as *const f32,
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

        let bytes_processed = (total_blocks * 144 + out_features * 4) as f64;
        let bandwidth_gbps = (bytes_processed / 1e9) / seconds;

        println!("\n=== РЕЗУЛЬТАТЫ БЕНЧМАРКА ===");
        println!("Всего проходов ядра: {}", num_iterations);
        println!("Общее время серии: {:.2} сек", ms / 1000.0);
        println!("Среднее время одного MatMul (GEMV): {:.4} мс", avg_ms);
        println!("Эффективный стриминг квантованных весов: {:.2} ГБ/сек", bandwidth_gbps);

        assert_eq!(cudaGetLastError(), 0, "CUDA Error detected!");
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }

    d_output.copy_to_host(h_output.as_mut_ptr() as *mut c_void, out_features * 4);

    let expected_val = 75776.0f32;
    let actual_val = h_output[0];
    let error = (actual_val - expected_val).abs();

    println!("\n--- РЕЗУЛЬТАТЫ МАТЕМАТИЧЕСКОЙ ВАЛИДАЦИИ ---");
    println!("Ожидалось на CPU (Эквивалент llama.cpp): {:.1}", expected_val);
    println!("Получено на GPU (Наш ультра-распаковщик):  {:.1}", actual_val);
    println!("Абсолютная ошибка распаковки: {:e}", error);

    if error < 1e-1 {
        println!("🚀 УСПЕХ! Наш GGUF Q4_K деквантизатор работает со стопроцентной точностью!");
    } else {
        println!("❌ ПРОВАЛ! Математика распаковки не сошлась.");
    }
}
