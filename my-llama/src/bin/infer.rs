use my_llama::models::kv_cache::KvCacheManager;
use my_llama::models::{KvCache, LlamaModel};
use my_llama::token::BpeTokenizer;
use my_llama::utils::safetensors::SafeTensorLoader;
use my_llama::utils::{CudaBuffer, CudaStream, PinnedHostBuffer};
use std::ffi::c_void;
use std::io::{stdout, Write};

unsafe extern "C" {
    fn launch_argmax(
        output_index: *mut i32,
        logits: *const f32,
        vocab_size: i32,
        stream: *mut c_void,
    );
}

fn main() -> std::io::Result<()> {
    println!("=================================================================");
    println!("    ЗАПУСК ПРОДАКШЕН ДВИЖКА ИНФЕРЕНСА (Qwen3.5-2B на GPU)        ");
    println!("=================================================================");

    my_llama::init_framework();

    const NUM_LAYERS: usize = 24;
    const HIDDEN_SIZE: usize = 2048;
    const NUM_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 2;
    const HIDDEN_FEATURES: usize = 5504;
    const VOCAB_SIZE: usize = 248320;
    const MAX_SEQ_LEN: usize = 512;

    let stream = CudaStream::new();

    let tokenizer = BpeTokenizer::from_file("tokenizer.json")
        .expect("Критическая ошибка: Не удалось загрузить оригинальный tokenizer.json для Qwen");

    let model = LlamaModel::new(
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HIDDEN_FEATURES,
        NUM_LAYERS,
        VOCAB_SIZE,
        &stream,
    );

    let loader = SafeTensorLoader::open("model.safetensors")?;

    println!("\n--- ЗАГРУЗКА ВЕСОВ МОДЕЛИ ---");
    model.load_weights(&loader, &stream)?;

    // Выделяем память под рабочие буферы
    let req_elements = model.required_workspace_elements(1, MAX_SEQ_LEN);
    let gpu_workspace = CudaBuffer::new(req_elements + HIDDEN_SIZE * 4);
    let gpu_logits = CudaBuffer::new(VOCAB_SIZE);
    let gpu_next_token_idx = CudaBuffer::new_int(1);

    let mut pinned_prompt_buffer = PinnedHostBuffer::new(MAX_SEQ_LEN);

    let kv_managers: Vec<KvCacheManager> = (0..NUM_LAYERS)
        .map(|_| KvCacheManager::new(MAX_SEQ_LEN, HIDDEN_SIZE))
        .collect();

    let prompt = "Привет, как твои дела?";
    println!("\nПользователь ввел: \"{}\"", prompt);

    let prompt_tokens_len = tokenizer.encode_to_pinned(prompt, &mut pinned_prompt_buffer);

    let mut input_ids = Vec::with_capacity(prompt_tokens_len);
    let pinned_slice = pinned_prompt_buffer.as_slice_mut();
    for i in 0..prompt_tokens_len {
        input_ids.push(pinned_slice[i] as u32);
    }
    println!("Токенизированные стартовые ID Qwen: {:?}", input_ids);

    println!("\n--- СТАРТ АВТОРЕГРЕССИВНОЙ ГЕНЕРАЦИИ ТОКЕНОВ НА ТЕНЗОРНЫХ ЯДРАХ ---");

    let mut generated_tokens = Vec::new();
    let mut host_token_result = vec![0; 1];

    {
        let mut kv_views: Vec<KvCache> = kv_managers.iter().map(|m| m.get_view()).collect();

        model.forward(
            &input_ids,
            &mut kv_views,
            &gpu_workspace,
            &gpu_logits,
            &stream,
        );

        unsafe {
            launch_argmax(
                gpu_next_token_idx.as_raw_ptr() as *mut i32,
                gpu_logits.as_raw_ptr() as *const f32,
                VOCAB_SIZE as i32,
                stream.as_raw(),
            );
        }

        gpu_next_token_idx.copy_to_host_async(&mut host_token_result, &stream);
        stream.synchronize();
    }

    let mut next_token_id = host_token_result[0] as u32;
    if next_token_id != tokenizer.eos_token_id {
        generated_tokens.push(next_token_id);
        let token_text = tokenizer.decode(&[next_token_id]);
        print!("{}", token_text);
        Write::flush(&mut stdout())?;
    }

    for _ in 0..30 {
        if next_token_id == tokenizer.eos_token_id {
            print!("[EOS]");
            break;
        }

        let current_step_input = vec![next_token_id];

        let mut kv_views: Vec<KvCache> = kv_managers.iter().map(|m| m.get_view()).collect();

        model.forward(
            &current_step_input,
            &mut kv_views,
            &gpu_workspace,
            &gpu_logits,
            &stream,
        );

        unsafe {
            launch_argmax(
                gpu_next_token_idx.as_raw_ptr() as *mut i32,
                gpu_logits.as_raw_ptr() as *const f32,
                VOCAB_SIZE as i32,
                stream.as_raw(),
            );
        }

        gpu_next_token_idx.copy_to_host_async(&mut host_token_result, &stream);
        stream.synchronize();

        next_token_id = host_token_result[0] as u32;

        if next_token_id == tokenizer.eos_token_id {
            print!("[EOS]");
            break;
        }

        generated_tokens.push(next_token_id);

        let token_text = tokenizer.decode(&[next_token_id]);
        print!("{}", token_text);
        Write::flush(&mut stdout())?;
    }

    println!("\n-----------------------------------------------------------------");
    println!("[УСПЕХ] Инференс реальной Qwen3.5-2B успешно выполнен на видеокарте!");
    println!("=================================================================");
    Ok(())
}
