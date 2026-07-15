use std::ffi::{c_char, c_void};
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_flash_decoding(
        output: *mut f32,
        partial_out: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        query: *const f32,
        k_cache: *const c_void,
        v_cache: *const c_void,
        k_scales: *const f32,
        v_scales: *const f32,
        cache_type_id: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        current_seq_len: i32,
        chunk_size: i32,
        stream: *mut c_void,
    );
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaStreamCreateWithFlags(p_stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;

    fn cudaStreamBeginCapture(stream: *mut c_void, flags: u32) -> i32;
    fn cudaStreamEndCapture(stream: *mut c_void, p_graph: *mut *mut c_void) -> i32;
    fn cudaGraphInstantiate(
        p_exec: *mut *mut c_void,
        graph: *mut c_void,
        p_error_node: *mut *mut c_void,
        log_buf: *mut c_char,
        log_buf_size: usize,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
}

struct CudaBuffer {
    ptr: *mut c_void,
    size_bytes: usize,
}

impl CudaBuffer {
    fn alloc(size_bytes: usize) -> Self {
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            assert_eq!(cudaMalloc(&mut raw_ptr, size_bytes), 0);
        }
        CudaBuffer { ptr: raw_ptr, size_bytes }
    }
    fn copy_to_device(&self, host_data: *const c_void, bytes: usize) {
        assert!(bytes <= self.size_bytes);
        unsafe {
            cudaMemcpy(self.ptr, host_data, bytes, 1);
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

fn simulate_decode_generation(
    type_id: i32,
    type_name: &str,
    bytes_per_elem: usize,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    start_seq_len: i32,
    gen_tokens_count: i32,
    chunk_size: i32,
    d_query: &CudaBuffer,
    d_output: &CudaBuffer,
    stream: *mut c_void,
) {
    let max_context = start_seq_len + gen_tokens_count;
    let max_num_chunks = (max_context + chunk_size - 1) / chunk_size;

    let kv_cache_size = (max_context * num_kv_heads * head_dim) as usize;
    let partial_out_size = (num_heads * max_num_chunks * head_dim) as usize;
    let partial_meta_size = (num_heads * max_num_chunks) as usize;

    let d_k_cache = CudaBuffer::alloc(kv_cache_size * bytes_per_elem);
    let d_v_cache = CudaBuffer::alloc(kv_cache_size * bytes_per_elem);

    let d_partial_out = CudaBuffer::alloc(partial_out_size * 4);
    let d_partial_max = CudaBuffer::alloc(partial_meta_size * 4);
    let d_partial_sum = CudaBuffer::alloc(partial_meta_size * 4);

    let h_scales = vec![1.0f32; (max_context * num_kv_heads) as usize];
    let d_k_scales = CudaBuffer::alloc(h_scales.len() * 4);
    let d_v_scales = CudaBuffer::alloc(h_scales.len() * 4);
    if type_id == 2 {
        d_k_scales.copy_to_device(h_scales.as_ptr() as *const c_void, h_scales.len() * 4);
        d_v_scales.copy_to_device(h_scales.as_ptr() as *const c_void, h_scales.len() * 4);
    }

    unsafe {
        cudaDeviceSynchronize();
    }

    let mut graph: *mut c_void = ptr::null_mut();
    let mut graph_exec: *mut c_void = ptr::null_mut();
    let mut total_bytes_processed: u64 = 0;

    unsafe {
        assert_eq!(cudaStreamBeginCapture(stream, 0), 0);

        for step in 0..gen_tokens_count {
            let current_len = start_seq_len + step;

            launch_flash_decoding(
                d_output.ptr as *mut f32,
                d_partial_out.ptr as *mut f32,
                d_partial_max.ptr as *mut f32,
                d_partial_sum.ptr as *mut f32,
                d_query.ptr as *const f32,
                d_k_cache.ptr as *const c_void,
                d_v_cache.ptr as *const c_void,
                d_k_scales.ptr as *const f32,
                d_v_scales.ptr as *const f32,
                type_id,
                num_heads,
                num_kv_heads,
                head_dim,
                current_len,
                chunk_size,
                stream,
            );

            let chunks_at_step = (current_len + chunk_size - 1) / chunk_size;
            let cache_bytes = (current_len * num_kv_heads * head_dim) as usize * bytes_per_elem * 2;
            let static_bytes = ((num_heads * head_dim) + (num_heads * head_dim)) as usize * 4;
            let partial_bytes = ((num_heads * chunks_at_step * head_dim) * 2 + (num_heads * chunks_at_step) * 4) as usize * 4;

            total_bytes_processed += (static_bytes + cache_bytes + partial_bytes) as u64;
        }

        assert_eq!(cudaStreamEndCapture(stream, &mut graph), 0);
        assert_eq!(cudaGraphInstantiate(&mut graph_exec, graph, ptr::null_mut(), ptr::null_mut(), 0), 0);
        cudaDeviceSynchronize();
    }

    let start_time = Instant::now();
    unsafe {
        assert_eq!(cudaGraphLaunch(graph_exec, stream), 0);
        assert_eq!(cudaDeviceSynchronize(), 0);
    }
    let elapsed_seconds = start_time.elapsed().as_secs_f64();

    let tokens_per_second = gen_tokens_count as f64 / elapsed_seconds;
    let avg_bandwidth_gbps = (total_bytes_processed as f64 / 1e9) / elapsed_seconds;

    println!(
        "| {:<6} | {:<20.2} | {:<22.2} | {:<14.5} |",
        type_name, avg_bandwidth_gbps, tokens_per_second, elapsed_seconds
    );

    unsafe {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
    }
}

fn main() {
    println!("=== РЕАЛИСТИЧНЫЙ СИМУЛЯТОР СЕССИИ ГЕНЕРАЦИИ (DECODE PAYLOAD) ===");

    let num_heads = 40;
    let num_kv_heads = 8;
    let head_dim = 128;
    let start_seq_len = 4096; // Стартуем с длинного промпта
    let gen_tokens_count = 500; // Эмулируем генерацию длинного ответа в 500 токенов
    let chunk_size = 256;

    println!("Параметры сессии:");
    println!("Модель: Qwen-35B геометрия, Базовый контекст: {} токенов", start_seq_len);
    println!(
        "Длина генерации ответа: {} новых токенов (Итоговый контекст: {})",
        gen_tokens_count,
        start_seq_len + gen_tokens_count
    );
    println!("Размер Split-K чанка: {}\n", chunk_size);

    let q_size = (num_heads * head_dim) as usize;
    let out_size = (num_heads * head_dim) as usize;
    let h_query = vec![1.0f32; q_size];

    let d_query = CudaBuffer::alloc(q_size * 4);
    let d_output = CudaBuffer::alloc(out_size * 4);
    d_query.copy_to_device(h_query.as_ptr() as *const c_void, q_size * 4);

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("+--------+----------------------+------------------------+----------------+");
        println!("| Format | Real Bandwidth (GB/s)| Speed (Tokens/Second)  | Total Time (s) |");
        println!("+--------+----------------------+------------------------+----------------+");

        // 1. Симулируем генерацию 500 токенов на тяжелом FP32 кэше
        simulate_decode_generation(
            0,
            "FP32",
            4,
            num_heads,
            num_kv_heads,
            head_dim,
            start_seq_len,
            gen_tokens_count,
            chunk_size,
            &d_query,
            &d_output,
            stream,
        );

        // 2. Симулируем генерацию 500 токенов на стандартном FP16 кэше
        simulate_decode_generation(
            1,
            "FP16",
            2,
            num_heads,
            num_kv_heads,
            head_dim,
            start_seq_len,
            gen_tokens_count,
            chunk_size,
            &d_query,
            &d_output,
            stream,
        );

        // 3. Симулируем генерацию 500 токенов на квантованном FP8 кэше
        simulate_decode_generation(
            2,
            "FP8",
            1,
            num_heads,
            num_kv_heads,
            head_dim,
            start_seq_len,
            gen_tokens_count,
            chunk_size,
            &d_query,
            &d_output,
            stream,
        );

        println!("+--------+----------------------+------------------------+----------------+");

        assert_eq!(cudaGetLastError(), 0);
        cudaStreamDestroy(stream);
    }
}
