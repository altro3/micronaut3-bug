use my_llama::models::block::TransformerBlock;
use my_llama::models::kv_cache::KvCacheManager;
use my_llama::token::BpeTokenizer;
use my_llama::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {
    // Подключаем ультимативную асинхронную сигнатуру ArgMax со стримом
    fn launch_argmax(
        output_index: *mut i32,
        logits: *const f32,
        vocab_size: i32,
        stream: *mut c_void,
    );
}

fn main() {
    println!("=== УЛЬТИМАТИВНЫЙ АСИНХРОННЫЙ ДВИЖОК ИНФЕРЕНСА (Tensor Cores) ===");

    // 1. Аппаратный прогрев cuBLAS
    my_llama::init_framework();

    const BATCH_SIZE: usize = 1;
    const HIDDEN_SIZE: usize = 4;
    const NUM_HEADS: usize = 2;
    const NUM_KV_HEADS: usize = 1;
    const HIDDEN_FEATURES: usize = 6;
    const MAX_SEQ_LEN: usize = 32;
    const VOCAB_SIZE: usize = 260;

    // Инициализируем неблокирующую очередь команд для GPU
    let stream = CudaStream::new();
    let tokenizer = BpeTokenizer::new_micro();

    let prompt = "привет";
    println!("Пользователь ввел: \"{}\"", prompt);

    let mut input_ids = tokenizer.encode(prompt);
    if input_ids.last() == Some(&tokenizer.eos_token_id) {
        input_ids.pop();
    }
    println!("Стартовые ID токенов: {:?}", input_ids);

    // =========================================================================
    // ВЫДЕЛЕНИЕ ПАМЯТИ ПРИ СТАРТЕ (0 аллокаций в рантайме)
    // =========================================================================
    let mut kv_manager = KvCacheManager::new(MAX_SEQ_LEN, HIDDEN_SIZE);
    let transformer_block = TransformerBlock::new(
        HIDDEN_SIZE,
        NUM_HEADS,
        NUM_KV_HEADS,
        HIDDEN_FEATURES,
        &stream,
    );

    // Рассчитываем и выделяем единый Workspace-черновик для внутренних слоев TransformerBlock
    let req_workspace_elements =
        transformer_block.required_workspace_elements(BATCH_SIZE, MAX_SEQ_LEN);
    let gpu_workspace = CudaBuffer::new(req_workspace_elements);

    // Постоянные буферы-переменные для генерации (выделены ДО горячего цикла)
    let current_x = CudaBuffer::new(HIDDEN_SIZE);
    let logits_buffer = CudaBuffer::new(VOCAB_SIZE);

    // Безопасный, обернутый в CudaBuffer интеджер под результат ArgMax
    let gpu_next_token_idx = CudaBuffer::new_int(1);

    let dummy_k = CudaBuffer::new(HIDDEN_SIZE);
    let dummy_v = CudaBuffer::new(HIDDEN_SIZE);

    // Имитируем заполнение кэша для стартового промпта
    dummy_k.copy_from_host_async(&vec![1.0f32; HIDDEN_SIZE], &stream);
    dummy_v.copy_from_host_async(&vec![1.0f32; HIDDEN_SIZE], &stream);
    for _ in 0..input_ids.len() {
        kv_manager.append_async(&dummy_k, &dummy_v, &stream);
    }

    println!("\n--- СТАРТ АВТОРЕГРЕССИВНОЙ ГЕНЕРАЦИИ ---");

    let mut generated_tokens = Vec::new();

    // Массивы-приемники на CPU для асинхронного считывания (0 аллокаций в цикле)
    let mut host_token_result = vec![0; 1];
    let mut mock_logits = vec![-100.0f32; VOCAB_SIZE];

    for step in 0..5 {
        let last_token = *input_ids.last().unwrap();

        // Асинхронно закидываем последний сгенерированный токен на вход
        current_x.copy_from_host_async(&vec![last_token as f32; HIDDEN_SIZE], &stream);

        // Получаем легковесный взгляд на заполненную часть кэша за 0 наносекунд
        let kv_view = kv_manager.get_view();

        // Прогоняем весь блок трансформера в асинхронном стриме без аллокаций
        transformer_block.forward(&current_x, &kv_view, BATCH_SIZE, &gpu_workspace, &stream);

        // Имитируем выходные логиты языковой модели (Mock логики классификации)
        let next_mock_id = match step {
            0 => 208, // 'п'
            1 => 159, // 'П'
            2 => 209, // 'р'
            3 => 128, // 'р' часть 2
            _ => tokenizer.eos_token_id,
        };

        // Зануляем старый топ-логит и выставляем новый
        if step > 0 {
            let prev_mock_id = match step - 1 {
                0 => 208,
                1 => 159,
                2 => 209,
                3 => 128,
                _ => 0,
            };
            mock_logits[prev_mock_id] = -100.0f32;
        }
        mock_logits[next_mock_id as usize] = 50.0f32;

        // Асинхронно загружаем логиты на GPU
        logits_buffer.copy_from_host_async(&mock_logits, &stream);

        // Запускаем асинхронный ArgMax для выбора лучшего токена в стриме
        unsafe {
            launch_argmax(
                gpu_next_token_idx.as_raw_ptr() as *mut i32,
                logits_buffer.as_raw_ptr() as *const f32,
                VOCAB_SIZE as i32,
                stream.as_raw(),
            );
        }

        // Асинхронно скачиваем ID выбранного токена обратно на хост
        gpu_next_token_idx.copy_to_host_async(&mut host_token_result, &stream);

        // Единственная точка синхронизации процессора за весь такт генерации!
        stream.synchronize();

        let next_token_id = host_token_result[0] as u32;

        if next_token_id == tokenizer.eos_token_id {
            break;
        }

        input_ids.push(next_token_id);
        generated_tokens.push(next_token_id);

        // Асинхронно обновляем KV-кэш контекста для следующего шага
        dummy_k.copy_from_host_async(&vec![0.5f32; HIDDEN_SIZE], &stream);
        dummy_v.copy_from_host_async(&vec![0.5f32; HIDDEN_SIZE], &stream);

        if !kv_manager.is_full() {
            kv_manager.append_async(&dummy_k, &dummy_v, &stream);
        }
    }

    let final_generated_text = tokenizer.decode(&generated_tokens);
    println!("{}{}[EOS]", prompt, final_generated_text);

    println!("\n--- ГЕНЕРАЦИЯ ЗАВЕРШЕНА ПО СИГНАЛУ [EOS] ---");
    // Больше никакого небезопасного cudaFree вручную! RAII полностью очистит буферы.
}
