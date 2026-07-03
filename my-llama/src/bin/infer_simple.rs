use my_llama::tokenizer::bpe_tokenizer::BpeTokenizer;
use my_llama::tokenizer::context::TokenizationContext;
use std::collections::HashMap;

fn main() {
    println!("=== ИНИЦИАЛИЗАЦИЯ ЧИСТОГО ТЕСТОВОГО СЛОВАРЯ BPE ===");

    let mut raw_pairs = Vec::with_capacity(64);
    let mut byte_fallback = [0u32; 256];
    let mut vocab_builder = HashMap::with_capacity(512);

    // ДОБАВЛЕНО: Буфер для упорядоченной компиляции обратного словаря
    // Выделяем с запасом (в тесте будет около 300 токенов, берем 512)
    let mut vocab_compiled_tokens = vec![Vec::new(); 512];

    // 1. Базовые байты (0..255)
    for b in 0..=255 {
        byte_fallback[b] = b as u32;
        let c = b as u8;

        // Для вывода в консоль
        let byte_string = String::from_utf8(vec![c]).unwrap_or_else(|_| format!("\\x{:02x}", b));
        vocab_builder.insert(b as u32, byte_string);

        // ДОБАВЛЕНО: Пишем сырые базовые байты в обратный вокабуляр по индексу ID
        vocab_compiled_tokens[b] = vec![c];
    }

    // 2. ВРУЧНУЮ собираем железную иерархию токенов для слова encode_parallel
    let mut current_id = 256u32;
    let mut current_rank = 0u32;

    let mut add_explicit_pair = |left: u32, right: u32, text_repr: &str| -> u32 {
        let pack = ((left as u64) << 32) | (right as u64);
        raw_pairs.push((pack, (current_rank, current_id)));
        vocab_builder.insert(current_id, text_repr.to_string());

        // ДОБАВЛЕНО: Кладем байты нового составного токена строго по его ID
        let id_idx = current_id as usize;
        if id_idx >= vocab_compiled_tokens.len() {
            vocab_compiled_tokens.resize(id_idx + 64, Vec::new());
        }
        vocab_compiled_tokens[id_idx] = text_repr.as_bytes().to_vec();

        let allocated_id = current_id;
        current_id += 1;
        current_rank += 1;
        allocated_id
    };

    // Строим "encode"
    let t_en = add_explicit_pair(b'e' as u32, b'n' as u32, "en");
    let t_enc = add_explicit_pair(t_en, b'c' as u32, "enc");
    let t_enco = add_explicit_pair(t_enc, b'o' as u32, "enco");
    let t_encod = add_explicit_pair(t_enco, b'd' as u32, "encod");
    let t_encode = add_explicit_pair(t_encod, b'e' as u32, "encode");

    // Строим "parallel"
    let t_pa = add_explicit_pair(b'p' as u32, b'a' as u32, "pa");
    let t_par = add_explicit_pair(t_pa, b'r' as u32, "par");
    let t_para = add_explicit_pair(t_par, b'a' as u32, "para");
    let t_paral = add_explicit_pair(t_para, b'l' as u32, "paral");
    let t_parall = add_explicit_pair(t_paral, b'l' as u32, "parall");
    let t_paralle = add_explicit_pair(t_parall, b'e' as u32, "paralle");
    let t_parallel = add_explicit_pair(t_paralle, b'l' as u32, "parallel");

    let t_encode_box = add_explicit_pair(t_encode, b'_' as u32, "encode_");
    // Финал: encode_ + parallel -> encode_parallel
    let _t_encode_parallel = add_explicit_pair(t_encode_box, t_parallel, "encode_parallel");

    let t_pu = add_explicit_pair(b'p' as u32, b'u' as u32, "pu");
    let _t_pub = add_explicit_pair(t_pu, b'b' as u32, "pub");
    let t_un = add_explicit_pair(b'u' as u32, b'n' as u32, "un");
    let t_uns = add_explicit_pair(t_un, b's' as u32, "uns");
    let t_unsa = add_explicit_pair(t_uns, b'a' as u32, "unsa");
    let t_unsaf = add_explicit_pair(t_unsa, b'f' as u32, "unsaf");
    let _t_unsafe = add_explicit_pair(t_unsaf, b'e' as u32, "unsafe");
    let _t_fn = add_explicit_pair(b'f' as u32, b'n' as u32, "fn");

    let vocab_size = current_id as usize;

    vocab_compiled_tokens.truncate(vocab_size);

    println!("Словарь собран. Чистый размер: {} токенов.", vocab_size);
    println!("Правил слияния BPE (raw_pairs): {}\n", raw_pairs.len());

    let tokenizer = BpeTokenizer::new(&raw_pairs, byte_fallback, current_id, vocab_size, &vocab_compiled_tokens);
    let mut ctx = TokenizationContext::new(vocab_size, 4096);

    let input_text = "pub unsafe fn encode_parallel";

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

    // ДОБАВЛЕНО ДЛЯ ПРОВЕРКИ: Тестируем рантайм-метод decode, который мы вынесли в decoder.rs!
    println!("\n=== ЗАПУСК ГИПЕРЗВУКОВОГО ДЕКОДЕРА ===");
    let decoded_output = tokenizer.decode(tokens);
    println!("Раскодированный текст: '{}'", decoded_output);
    assert_eq!(input_text, decoded_output, "КРИТИЧЕСКИЙ СБОЙ: Итоговый текст раскодирован с ошибками!");
    println!("Верификация успешна: Обратное декодирование совпало на 100%!");

    println!("\nСтатистика:");
    println!("Исходный размер : {} байт", input_text.len());
    println!("Всего токенов   : {}", tokens.len());
    println!("Коэффициент     : {:.2}x сжатие текста", input_text.len() as f32 / tokens.len() as f32);
    println!("Время работы    : {:?}", duration);
    println!("===========================================================");
}
