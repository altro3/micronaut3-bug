pub mod models;
pub mod token;
pub mod utils;

use models::KvCache;
use models::TransformerBlock;
use std::ffi::c_void;
use token::BpeTokenizer;
use utils::CudaBuffer;

unsafe extern "C" {
    fn launch_argmax(output_index: *mut i32, logits: *const c_void, vocab_size: i32);
}

fn main() {
    // Жесткий хак для Windows PowerShell: Переключаем терминал в режим UTF-8
    #[cfg(target_os = "windows")]
    if let Err(e) = std::process::Command::new("cmd")
        .args(&["/C", "chcp 65001"])
        .output()
    {
        eprintln!(
            "Предупреждение: Не удалось переключить консоль в UTF-8: {:?}",
            e
        );
    }

    println!("=== ЗАПУСК ДВИЖКА ИНФЕРЕНСА (GENERATION LOOP) MY-QWEN ===");

    // 1. Конфигурация модели и словаря
    const HIDDEN_SIZE: usize = 4;
    const NUM_HEADS: usize = 2;
    const NUM_KV_HEADS: usize = 1;
    const HIDDEN_FEATURES: usize = 6;
    const MAX_SEQ_LEN: usize = 32;
    const VOCAB_SIZE: usize = 260;

    // Инициализируем наш восстановленный микро-токенизатор
    let tokenizer = BpeTokenizer::new_micro();

    let prompt = "привет";
    println!("Пользователь ввел: \"{}\"", prompt);

    let mut input_ids = tokenizer.encode(prompt);
    if input_ids.last() == Some(&tokenizer.eos_token_id) {
        input_ids.pop();
    }
    println!("Стартовые ID токенов: {:?}", input_ids);

    // 2. Аллокация памяти во VRAM
    let mut kv_cache = KvCache::new(MAX_SEQ_LEN, HIDDEN_SIZE);
    let transformer_block =
        TransformerBlock::new(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS, HIDDEN_FEATURES);
    let d_output_token_idx = CudaBuffer::new_int_scalar();

    // Предзаполняем KV-Cache стартовым контекстом
    for _ in 0..input_ids.len() {
        let dummy_k = CudaBuffer::new(HIDDEN_SIZE);
        let dummy_v = CudaBuffer::new(HIDDEN_SIZE);
        dummy_k.copy_from_host(&vec![1.0f32; HIDDEN_SIZE]);
        dummy_v.copy_from_host(&vec![1.0f32; HIDDEN_SIZE]);
        kv_cache.append(&dummy_k, &dummy_v);
    }

    println!("\n--- СТАРТ АВТОРЕГРЕССИВНОЙ ГЕНЕРАЦИИ ---");

    let mut generated_tokens = Vec::new();

    for step in 0..5 {
        let current_x = CudaBuffer::new(HIDDEN_SIZE);
        let last_token = *input_ids.last().unwrap();
        current_x.copy_from_host(&vec![last_token as f32; HIDDEN_SIZE]);

        transformer_block.forward(&current_x, &mut kv_cache, 1);

        let logits_buffer = CudaBuffer::new(VOCAB_SIZE);
        let mut mock_logits = vec![-100.0f32; VOCAB_SIZE];

        let next_mock_id = match step {
            0 => 208,                    // 'п'
            1 => 159,                    // 'П'
            2 => 209,                    // 'р'
            3 => 128,                    // 'р' часть 2
            _ => tokenizer.eos_token_id, // Легитимный токен останова [EOS] (256)
        };
        mock_logits[next_mock_id as usize] = 50.0f32;
        logits_buffer.copy_from_host(&mock_logits);

        unsafe {
            launch_argmax(
                d_output_token_idx as *mut i32,
                logits_buffer.as_raw_ptr(),
                VOCAB_SIZE as i32,
            );
        }

        let next_token_id = CudaBuffer::copy_int_to_host(d_output_token_idx) as u32;

        if next_token_id == tokenizer.eos_token_id {
            break;
        }

        input_ids.push(next_token_id);
        generated_tokens.push(next_token_id);

        let dummy_k = CudaBuffer::new(HIDDEN_SIZE);
        let dummy_v = CudaBuffer::new(HIDDEN_SIZE);
        dummy_k.copy_from_host(&vec![0.5f32; HIDDEN_SIZE]);
        dummy_v.copy_from_host(&vec![0.5f32; HIDDEN_SIZE]);
        if !kv_cache.is_full() {
            kv_cache.append(&dummy_k, &dummy_v);
        }
    }

    // Выводим результат: декодируем всю пачку сгенерированных байт разом
    let final_generated_text = tokenizer.decode(&generated_tokens);
    println!("{}{}[EOS]", prompt, final_generated_text);

    println!("\n--- ГЕНЕРАЦИЯ ЗАВЕРШЕНА ПО СИГНАЛУ [EOS] ---");

    unsafe {
        unsafe extern "C" {
            fn cudaFree(p: *mut c_void) -> i32;
        }
        cudaFree(d_output_token_idx);
    }
}
