pub mod utils;
pub mod token;
pub mod models;

use std::ffi::c_void;
use utils::CudaBuffer;
use token::BpeTokenizer;

unsafe extern "C" {
    fn test_cuda_setup(d_array: *mut c_void, size: i32);
}

fn main() {
    println!("=== МИДЛ-ТЕСТ: ЧИСТАЯ АРХИТЕКТУРА MY-Qwen ===");

    let tokenizer = BpeTokenizer::new_micro();
    // Тестируем строку с пробелом. Пробел больше не потеряется!
    let text_prompt = "привет привет";

    let token_ids = tokenizer.encode(text_prompt);
    println!("ID токенов: {:?}", token_ids);

    let decoded_text = tokenizer.decode(&token_ids);
    println!("Восстановленный текст: \"{}\"", decoded_text);

    // Работа с GPU
    let context_size = token_ids.len();
    let gpu_buffer = CudaBuffer::new(context_size);
    let host_data: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();

    gpu_buffer.copy_from_host(&host_data);
    unsafe {
        test_cuda_setup(gpu_buffer.as_raw_ptr(), context_size as i32);
    }

    let result_from_gpu = gpu_buffer.copy_to_host();
    println!("Результат работы GPU: {:?}", result_from_gpu);
}
