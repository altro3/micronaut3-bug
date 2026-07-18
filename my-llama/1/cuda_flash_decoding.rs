use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_paged_flash_decoding_write(
        k_block_table: *mut c_void,
        v_block_table: *mut c_void,
        k_src: *const f32,
        v_src: *const f32,
        block_mapping: *const i32,
        seq_lengths: *const i32,
        k_scales: *mut f32,
        v_scales: *mut f32,
        cache_type_id: i32,
        num_seqs: i32,
        num_kv_heads: i32,
        head_dim: i32,
        max_blocks_per_seq: i32,
        block_size: i32,
        is_prefill: i32,
        stream: *mut c_void,
    );

    pub fn launch_paged_flash_decoding(
        output: *mut f32,
        partial_out: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        query: *const f32,
        k_block_table: *const c_void,
        v_block_table: *const c_void,
        block_mapping: *const i32,
        seq_lengths: *const i32,
        k_scales: *const f32,
        v_scales: *const f32,
        cache_type_id: i32,
        num_seqs: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        max_blocks_per_seq: i32,
        block_size: i32,
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
        log_buf: *mut std::ffi::c_char,
        log_buf_size: usize,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut c_void) -> i32;
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

fn simulate_paged_decode_generation(
    type_id: i32,
    type_name: &str,
    bytes_per_elem: usize,
    num_seqs: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    start_seq_len: i32,
    gen_tokens_count: i32,
    block_size: i32,
    chunk_size: i32,
    d_query: &CudaBuffer,
    d_output: &CudaBuffer,
    stream: *mut c_void,
) {
    let max_context = start_seq_len + gen_tokens_count;
    let max_blocks_per_seq = (max_context + block_size - 1) / block_size;
    let total_needed_blocks = (max_blocks_per_seq * num_seqs) as usize;
    let max_num_chunks = (max_context + chunk_size - 1) / chunk_size;

    let out_size = (num_seqs * num_heads * head_dim) as usize;
    let partial_out_size = (num_seqs * num_heads * max_num_chunks * head_dim) as usize;
    let partial_meta_size = (num_seqs * num_heads * max_num_chunks) as usize;

    let single_block_elements = (block_size * num_kv_heads * head_dim) as usize;
    let total_cache_elements = total_needed_blocks * single_block_elements;
    let cache_alloc_size = (total_cache_elements * bytes_per_elem + 3) / 4;

    let d_k_block_table = CudaBuffer::alloc(cache_alloc_size);
    let d_v_block_table = CudaBuffer::alloc(cache_alloc_size);

    let d_partial_out = CudaBuffer::alloc(partial_out_size);
    let d_partial_max = CudaBuffer::alloc(partial_meta_size);
    let d_partial_sum = CudaBuffer::alloc(partial_meta_size);

    let total_scale_elements = total_needed_blocks * (block_size * num_kv_heads) as usize;
    let d_k_scales = CudaBuffer::alloc(total_scale_elements);
    let d_v_scales = CudaBuffer::alloc(total_scale_elements);

    let h_dummy_cache = vec![0_u8; cache_alloc_size * 4];
    unsafe {
        cudaMemcpy(
            d_k_block_table.ptr as *mut c_void,
            h_dummy_cache.as_ptr() as *const c_void,
            cache_alloc_size * 4,
            1,
        );
        cudaMemcpy(
            d_v_block_table.ptr as *mut c_void,
            h_dummy_cache.as_ptr() as *const c_void,
            cache_alloc_size * 4,
            1,
        );
    }

    let mut h_block_mapping = vec![-1_i32; (num_seqs * max_blocks_per_seq) as usize];
    let mut block_id_counter = 0_i32;
    for s in 0..num_seqs {
        for b in 0..max_blocks_per_seq {
            h_block_mapping[(s * max_blocks_per_seq + b) as usize] = block_id_counter;
            block_id_counter += 1;
        }
    }
    let d_block_mapping = CudaBuffer::alloc(h_block_mapping.len());
    unsafe {
        cudaMemcpy(
            d_block_mapping.ptr as *mut c_void,
            h_block_mapping.as_ptr() as *const c_void,
            h_block_mapping.len() * 4,
            1,
        );
    }

    let d_k_src = CudaBuffer::alloc((num_seqs * num_kv_heads * head_dim) as usize);
    let d_v_src = CudaBuffer::alloc((num_seqs * num_kv_heads * head_dim) as usize);
    let h_src_dummy = vec![1.0_f32; d_k_src.size];
    d_k_src.copy_to_device(&h_src_dummy);
    d_v_src.copy_to_device(&h_src_dummy);

    let mut h_seq_lengths = vec![start_seq_len; num_seqs as usize];
    let d_seq_lengths = CudaBuffer::alloc(h_seq_lengths.len());

    unsafe {
        cudaDeviceSynchronize();
    }

    let mut graph: *mut c_void = ptr::null_mut();
    let mut graph_exec: *mut c_void = ptr::null_mut();
    let mut total_bytes_processed: u64 = 0;

    unsafe {
        assert_eq!(cudaStreamBeginCapture(stream, 0), 0);

        for step in 0..gen_tokens_count {
            for l in h_seq_lengths.iter_mut() {
                *l = start_seq_len + step + 1;
            }
            cudaMemcpy(
                d_seq_lengths.ptr as *mut c_void,
                h_seq_lengths.as_ptr() as *const c_void,
                h_seq_lengths.len() * 4,
                1,
            );

            launch_paged_flash_decoding_write(
                d_k_block_table.ptr as *mut c_void,
                d_v_block_table.ptr as *mut c_void,
                d_k_src.ptr,
                d_v_src.ptr,
                d_block_mapping.ptr as *const i32,
                d_seq_lengths.ptr as *const i32,
                d_k_scales.ptr,
                d_v_scales.ptr,
                type_id,
                num_seqs,
                num_kv_heads,
                head_dim,
                max_blocks_per_seq,
                block_size,
                0,
                stream,
            );

            launch_paged_flash_decoding(
                d_output.ptr,
                d_partial_out.ptr,
                d_partial_max.ptr,
                d_partial_sum.ptr,
                d_query.ptr,
                d_k_block_table.ptr as *const c_void,
                d_v_block_table.ptr as *const c_void,
                d_block_mapping.ptr as *const i32,
                d_seq_lengths.ptr as *const i32,
                d_k_scales.ptr,
                d_v_scales.ptr,
                type_id,
                num_seqs,
                num_heads,
                num_kv_heads,
                head_dim,
                start_seq_len + step + 1,
                max_blocks_per_seq,
                block_size,
                chunk_size,
                stream,
            );

            let current_len = start_seq_len + step + 1;
            let chunks_at_step = (current_len + chunk_size - 1) / chunk_size;

            let write_bytes =
                (num_seqs * num_kv_heads * head_dim) as usize * 4 * 2 + (num_seqs * num_kv_heads * head_dim) as usize * bytes_per_elem * 2;

            let cache_read_bytes = (num_seqs * chunks_at_step * chunk_size * num_kv_heads * head_dim) as usize * bytes_per_elem * 2;
            let static_bytes = (num_seqs * num_heads * head_dim * 2) as usize * 4;
            let partial_bytes =
                (num_seqs * num_heads * chunks_at_step * head_dim * 2) as usize * 4 + (num_seqs * num_heads * chunks_at_step * 2) as usize * 4;

            total_bytes_processed += (write_bytes + cache_read_bytes + static_bytes + partial_bytes) as u64;
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

    let mut h_output = vec![0.0f32; out_size];
    d_output.copy_to_host(&mut h_output);

    let validation_status = if h_output[0].is_finite() { "OK ✅" } else { "FAIL ❌" };

    println!(
        "| {:<6} | {:<20.2} | {:<22.2} | {:<14.5} | {:<10} |",
        type_name, avg_bandwidth_gbps, tokens_per_second, elapsed_seconds, validation_status
    );

    unsafe {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
    }
}

fn main() {
    println!("=== СИМУЛЯТОР СЕССИИ CONTINUOUS BATCHING С PAGED ATTENTION ===");

    let num_seqs = 4;
    let num_heads = 40;
    let num_kv_heads = 8;
    let head_dim = 128;
    let start_seq_len = 4096;
    let gen_tokens_count = 500;
    let block_size = 16;
    let chunk_size = 256;

    println!("Параметры сессии:");
    println!("Размер батча (num_seqs): {} запроса одновременно", num_seqs);
    println!("Модель: Qwen-35B геометрия, Базовый контекст: {} токенов", start_seq_len);
    println!("Длина генерации ответа: {} новых токенов", gen_tokens_count);
    println!("Размер Paged-блока: {} токенов, Размер Split-K чанка: {}\n", block_size, chunk_size);

    let q_size = (num_seqs * num_heads * head_dim) as usize;
    let out_size = (num_seqs * num_heads * head_dim) as usize;

    let d_query = CudaBuffer::alloc(q_size);
    let d_output = CudaBuffer::alloc(out_size);

    let h_query = vec![1.0f32; q_size];
    d_query.copy_to_device(&h_query);

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        println!("+--------+----------------------+------------------------+----------------+------------+");
        println!("| Format | Real Bandwidth (GB/s)| Speed (Tokens/Second)  | Total Time (s) | Validation |");
        println!("+--------+----------------------+------------------------+----------------+------------+");

        simulate_paged_decode_generation(0, "FP32", 4, num_seqs, num_heads, num_kv_heads, head_dim, start_seq_len, gen_tokens_count, block_size, chunk_size, &d_query, &d_output, stream);
        simulate_paged_decode_generation(1, "FP16", 2, num_seqs, num_heads, num_kv_heads, head_dim, start_seq_len, gen_tokens_count, block_size, chunk_size, &d_query, &d_output, stream);
        simulate_paged_decode_generation(2, "FP8", 1, num_seqs, num_heads, num_kv_heads, head_dim, start_seq_len, gen_tokens_count, block_size, chunk_size, &d_query, &d_output, stream);

        println!("+--------+----------------------+------------------------+----------------+------------+");

        assert_eq!(cudaGetLastError(), 0);
        cudaStreamDestroy(stream);
    }
}
