use my_llama::tokenizer::bpe_tokenizer::BpeTokenizer;
use my_llama::tokenizer::context::TokenizationContext;
use std::collections::HashMap;

fn main() {
    println!("=== ГЕНЕРАЦИЯ РЕАЛИСТИЧНОГО СЛОВАРЯ (10,000+ токенов) ===");

    let mut raw_pairs = Vec::new();
    let mut byte_fallback = [0u32; 256];
    let mut vocab_builder = HashMap::new();

    // 1. Инициализируем базовые байты ПРАВИЛЬНО: сохраняем РЕАЛЬНЫЕ символы в виде строк
    for b in 0..=255 {
        byte_fallback[b] = b as u32;
        let c = b as u8;

        // Мапим байт строго в его текстовый символ, чтобы при склейке получались слова
        let byte_string = String::from_utf8(vec![c])
            .unwrap_or_else(|_| format!("\\x{:02x}", b));

        vocab_builder.insert(b as u32, byte_string);
    }

    let mut current_token_id = 256u32;
    let mut current_rank = 0u32;

    let sub_words = [
        "unsafe", "return", "fn", "impl", "pub", "struct", "let", "mut", "match", "loop", "while",
        "println!", "core", "tokenizer", "context", "buffer", "capacity", "get_unchecked", "assert_eq!",
        "hello", "world", "parallel", "thread", "atomic", "Relaxed", "Ordering", "nvidia", "cuda", "alloc",
        "the", "and", "ing", "ion", "ent", "for", "that", "this", "with", "from", "_ptr", "_len", "idx",
        "0x", "ff", "::", "->", "=>", " {", "};", "();", "], ", "unsafe {\n", "std::", "Result<", "Vec<",
        "encode",
    ];

    let mut add_word_tokens = |word: &str| {
        let bytes = word.as_bytes();
        if bytes.len() < 2 { return; }

        let mut current_ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        loop {
            let mut merged_any = false;
            let mut i = 0;

            while i + 1 < current_ids.len() {
                let left = current_ids[i];
                let right = current_ids[i + 1];
                let pack = ((left as u64) << 32) | (right as u64);

                let existing = raw_pairs.iter().find(|&(p, _)| *p == pack).map(|&(_, (_, id))| id);

                if let Some(target_id) = existing {
                    current_ids[i] = target_id;
                    current_ids.remove(i + 1);
                    merged_any = true;
                } else {
                    raw_pairs.push((pack, (current_rank, current_token_id)));

                    let left_str = vocab_builder.get(&left).cloned().unwrap_or_default();
                    let right_str = vocab_builder.get(&right).cloned().unwrap_or_default();
                    vocab_builder.insert(current_token_id, format!("{}{}", left_str, right_str));

                    current_ids[i] = current_token_id;
                    current_ids.remove(i + 1);

                    current_token_id += 1;
                    current_rank += 1;
                    merged_any = true;
                }
            }

            if !merged_any { break; }
        }
    };

    for &w1 in sub_words.iter() {
        add_word_tokens(w1);
        for &w2 in sub_words.iter() {
            let combined = format!("{}{}", w1, w2);
            add_word_tokens(&combined);
        }
    }

    let vocab_size = current_token_id as usize;
    println!("Словарь собран. Размер словаря: {} токенов.", vocab_size);
    println!("Правил слияния BPE (raw_pairs): {}\n", raw_pairs.len());

    let tokenizer = BpeTokenizer::new(&raw_pairs, byte_fallback, current_token_id, vocab_size);
    let mut ctx = TokenizationContext::new(vocab_size, 4096);

    // Боевой текст
    let input_text = r#"
    pub unsafe fn encode_parallel(&self, texts: &[String]) -> Vec<Vec<u32>> {
        let mut ctx = TokenizationContext::new(self.vocab_size, optimal_chunk_capacity);
        loop {
            let idx = task_index.fetch_add(1, Ordering::Relaxed);
            if idx >= total_texts { break; }
            let text = unsafe { texts.get_unchecked(idx) };
            println!("hello world from nvidia cuda tokenizer core!");
            assert_eq!(ctx.tokens_buffer.capacity(), 4096);
        }
    }
    "#;

    println!("=== ЗАПУСК ТОКЕНИЗАЦИИ БОЕВОГО ТЕКСТА ===");
    println!("Длина текста: {} байт", input_text.len());

    let start_time = std::time::Instant::now();
    let tokens = tokenizer.encode(input_text, &mut ctx);
    let duration = start_time.elapsed();

    println!("\nВывод токенов в консоль:");
    println!("┌──────────┬─────────────┬──────────────────────────────────┐");
    println!("│ Индекс   │ ID токена   │ Восстановленный фрагмент текста  │");
    println!("├──────────┼─────────────┼──────────────────────────────────┤");

    for (i, &token_id) in tokens.iter().enumerate() {
        let token_str = vocab_builder.get(&token_id).cloned().unwrap_or_else(|| format!("ID_{}", token_id));
        let clean_str = token_str.replace("\n", "\\n").replace("\r", "\\r");
        println!("│ {:<8} │ {:<11} │ {:<32} │", i, token_id, clean_str);
    }
    println!("└──────────┴─────────────┴──────────────────────────────────┘");

    println!("\nСтатистика:");
    println!("Исходный размер : {} байт", input_text.len());
    println!("Всего токенов   : {}", tokens.len());
    println!("Коэффициент     : {:.2}x сжатие текста", input_text.len() as f32 / tokens.len() as f32);
    println!("Время работы    : {:?}", duration);

    let speed_gb_sec = (input_text.len() as f64 / 1024.0 / 1024.0 / 1024.0) / duration.as_secs_f64();
    println!("Скорость        : {:.2} ГБ/сек", speed_gb_sec);
    println!("===========================================================");
}
