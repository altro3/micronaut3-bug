use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[allow(clippy::duplicated_attributes)]
#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_speculative_verify(
        accepted_tokens: *mut i32,
        num_accepted: *mut i32,
        target_logits: *const f32,
        draft_probs: *const f32,
        draft_tokens: *const i32,
        random_nums: *const f32,
        workspace: *mut f32,
        num_seqs: i32,
        vocab_size: i32,
        max_draft_tokens: i32,
        temperature: f32,
        num_threads: i32,
        stream: *mut c_void,
    );

    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
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
    ptr: *mut c_void,
    size_bytes: usize,
}

impl CudaBuffer {
    fn alloc(size_bytes: usize) -> Self {
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut raw_ptr, size_bytes);
            assert_eq!(res, 0);
        }
        CudaBuffer { ptr: raw_ptr, size_bytes }
    }

    fn copy_to_device(&self, host_ptr: *const c_void, size_bytes: usize) {
        assert!(size_bytes <= self.size_bytes);
        unsafe {
            cudaMemcpy(self.ptr, host_ptr, size_bytes, 1);
        }
    }

    fn copy_to_host(&self, host_ptr: *mut c_void, size_bytes: usize) {
        assert!(size_bytes <= self.size_bytes);
        unsafe {
            cudaMemcpy(host_ptr, self.ptr, size_bytes, 2);
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
    println!("=== УЛЬТИМАТИВНЫЙ СТРЕСС-БЕНЧМАРК И ВАЛИДАЦИЯ SPECULATIVE VERIFY ===");

    let num_seqs = 2;
    let vocab_size = 152064;
    let max_draft_tokens = 4;
    let temperature = 0.7_f32;
    let num_threads = 1024;

    println!("Боевые параметры спецификации:");
    println!(
        "Batch (num_seqs): {}, Vocab Size: {}, Draft Window: {}, Temp: {}, Threads: {}",
        num_seqs, vocab_size, max_draft_tokens, temperature, num_threads
    );

    let target_logits_size = (num_seqs * max_draft_tokens * vocab_size) as usize;
    let mut h_target_logits = vec![-2.0f32; target_logits_size];

    let draft_meta_size = (num_seqs * max_draft_tokens) as usize;
    let mut h_draft_probs = vec![0.05f32; draft_meta_size];
    let mut h_draft_tokens = vec![0_i32; draft_meta_size];
    let mut h_random_nums = vec![0.0f32; (num_seqs * (max_draft_tokens + 1)) as usize];

    for step in 0..max_draft_tokens as usize {
        let idx = 0 * max_draft_tokens as usize + step;
        h_draft_tokens[idx] = (100 + step) as i32;
        h_draft_probs[idx] = 0.9f32;
        h_random_nums[idx] = 0.01f32;

        let logit_base = (0 * max_draft_tokens as usize + step) * vocab_size as usize;
        h_target_logits[logit_base + (100 + step)] = 20.0f32;
    }
    h_random_nums[0 * max_draft_tokens as usize + max_draft_tokens as usize] = 0.5f32;

    for step in 0..max_draft_tokens as usize {
        let idx = 1 * max_draft_tokens as usize + step;
        h_draft_tokens[idx] = (200 + step) as i32;
        h_draft_probs[idx] = 0.9f32;

        if step == 2 {
            h_random_nums[idx] = 0.99f32;
        } else {
            h_random_nums[idx] = 0.01f32;
        }

        let logit_base = (1 * max_draft_tokens as usize + step) * vocab_size as usize;
        if step == 2 {
            h_target_logits[logit_base + 777] = 25.0f32;
        } else {
            h_target_logits[logit_base + (200 + step)] = 20.0f32;
        }
    }
    h_random_nums[1 * max_draft_tokens as usize + max_draft_tokens as usize] = 0.0001f32;

    let out_tokens_size = (num_seqs * (max_draft_tokens + 1)) as usize;
    let mut h_accepted_tokens = vec![-1_i32; out_tokens_size];
    let mut h_num_accepted = vec![0_i32; num_seqs as usize];

    let d_workspace = CudaBuffer::alloc((num_seqs * vocab_size) as usize * 4);
    let d_target_logits = CudaBuffer::alloc(target_logits_size * 4);
    let d_draft_probs = CudaBuffer::alloc(draft_meta_size * 4);
    let d_draft_tokens = CudaBuffer::alloc(draft_meta_size * 4);
    let d_random_nums = CudaBuffer::alloc(h_random_nums.len() * 4);
    let d_accepted_tokens = CudaBuffer::alloc(out_tokens_size * 4);
    let d_num_accepted = CudaBuffer::alloc((num_seqs * 4) as usize);

    d_target_logits.copy_to_device(h_target_logits.as_ptr() as *const c_void, target_logits_size * 4);
    d_draft_probs.copy_to_device(h_draft_probs.as_ptr() as *const c_void, draft_meta_size * 4);
    d_draft_tokens.copy_to_device(h_draft_tokens.as_ptr() as *const c_void, draft_meta_size * 4);
    d_random_nums.copy_to_device(h_random_nums.as_ptr() as *const c_void, h_random_nums.len() * 4);

    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    unsafe {
        let mut stream: *mut c_void = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 0x01), 0);

        for _ in 0..NUM_WARMUP {
            launch_speculative_verify(
                d_accepted_tokens.ptr as *mut i32,
                d_num_accepted.ptr as *mut i32,
                d_target_logits.ptr as *const f32,
                d_draft_probs.ptr as *const f32,
                d_draft_tokens.ptr as *const i32,
                d_random_nums.ptr as *const f32,
                d_workspace.ptr as *mut f32,
                num_seqs,
                vocab_size,
                max_draft_tokens,
                temperature,
                num_threads,
                stream,
            );
        }
        cudaDeviceSynchronize();

        let mut graph: *mut c_void = ptr::null_mut();
        let mut graph_exec: *mut c_void = ptr::null_mut();

        assert_eq!(cudaStreamBeginCapture(stream, 0), 0);
        for _ in 0..NUM_ITERATIONS {
            launch_speculative_verify(
                d_accepted_tokens.ptr as *mut i32,
                d_num_accepted.ptr as *mut i32,
                d_target_logits.ptr as *const f32,
                d_draft_probs.ptr as *const f32,
                d_draft_tokens.ptr as *const i32,
                d_random_nums.ptr as *const f32,
                d_workspace.ptr as *mut f32,
                num_seqs,
                vocab_size,
                max_draft_tokens,
                temperature,
                num_threads,
                stream,
            );
        }
        assert_eq!(cudaStreamEndCapture(stream, &mut graph), 0);
        assert_eq!(cudaGraphInstantiate(&mut graph_exec, graph, ptr::null_mut(), ptr::null_mut(), 0), 0);
        cudaDeviceSynchronize();

        let start_host = Instant::now();
        assert_eq!(cudaGraphLaunch(graph_exec, stream), 0);
        assert_eq!(cudaDeviceSynchronize(), 0);
        let total_graph_time = start_host.elapsed();

        let avg_lat_us = (total_graph_time.as_secs_f64() * 1e6) / NUM_ITERATIONS as f64;
        let bytes_processed =
            (target_logits_size * 4 + draft_meta_size * 8 + h_random_nums.len() * 4 + out_tokens_size * 4 + (num_seqs * 4) as usize) as f64;
        let avg_bandwidth_gbps = (bytes_processed * NUM_ITERATIONS as f64 / 1e9) / total_graph_time.as_secs_f64();

        println!("\n📊 === РЕЗУЛЬТАТЫ АППАРАТНОГО ТЕСТА ЧЕРЕЗ CUDA GRAPHS ===");
        println!("Полное время выполнения 1000 итераций на GPU: {:.4} сек", total_graph_time.as_secs_f32());
        println!("Истинное среднее время валидации одного батча: {:.2} us", avg_lat_us);
        println!("Реальная утилизация шины памяти Blackwell:    {:.2} ГБ/сек", avg_bandwidth_gbps);
        println!("-------------------------------------------------------");

        d_accepted_tokens.copy_to_host(h_accepted_tokens.as_mut_ptr() as *mut c_void, out_tokens_size * 4);
        d_num_accepted.copy_to_host(h_num_accepted.as_mut_ptr() as *mut c_void, (num_seqs * 4) as usize);

        println!("\n--- ВАЛИДАЦИЯ ТОЧНОСТИ МАТЕМАТИКИ ЯДРА ---");

        println!("Запрос 0 (Ожидается полный успех): Accepted Count = 5, Tokens = [100, 101, 102, 103, 103]");
        print!("Фактически на GPU: Count = {}, Tokens = [", h_num_accepted[0]);
        for step in 0..max_draft_tokens as usize {
            print!("{}, ", h_accepted_tokens[0 * (max_draft_tokens + 1) as usize + step]);
        }
        println!("{}]", h_accepted_tokens[0 * (max_draft_tokens + 1) as usize + max_draft_tokens as usize]);

        println!("\nЗапрос 1 (Ожидается обрыв на шаге 2 + сэмплинг лидером 777): Accepted Count = 2, Tokens = [200, 201, 777, 0, 0]");
        print!("Фактически на GPU: Count = {}, Tokens = [", h_num_accepted[1]);
        for step in 0..max_draft_tokens as usize {
            print!("{}, ", h_accepted_tokens[1 * (max_draft_tokens + 1) as usize + step]);
        }
        println!("{}]", h_accepted_tokens[1 * (max_draft_tokens + 1) as usize + max_draft_tokens as usize]);

        let valid_0 = h_num_accepted[0] == 5 && h_accepted_tokens[0 * 5 + 0] == 100 && h_accepted_tokens[0 * 5 + 3] == 103;
        let valid_1 = h_num_accepted[1] == 2 && h_accepted_tokens[1 * 5 + 2] == 777;

        if valid_0 && valid_1 {
            println!("\n🎉 ПОБЕДА! Спекулятивное ядро идеально прошло стресс-тесты и верно обрабатывает как совпадения, так и обрывы цепочек!");
        } else {
            println!("\n❌ МАТЕМАТИЧЕСКИЙ СБОЙ: Логика верификации или резервного сэмплирования нарушена.");
        }
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
}
