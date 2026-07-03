#[cfg(test)]
mod tests {
    use my_llama::tokenizer::SimdSplitter;

    fn create_ai_fallback_table() -> [u32; 256] {
        let mut fallback = [0u32; 256];
        for i in 0..256 {
            fallback[i] = i as u32;
        }
        fallback
    }

    #[test]
    fn test_russian_prompt_strict_chunks_validation() {
        let byte_fallback = create_ai_fallback_table();

        let russian_prompt = "Привет! Ты — мощная языковая модель Llama.\n\
    Твоя задача — обрабатывать огромные объемы текста на русском языке.\t\
    SIMD-алгоритмы помогают делать это на космической скорости в 2026 году!";

        let mut simd_ids = vec![0u32; russian_prompt.len()];
        let simd_count = SimdSplitter::split(russian_prompt, &mut simd_ids, &byte_fallback);
        let final_simd_ids = &simd_ids[..simd_count];

        // 1. Переводим результат SIMD в байты
        let simd_bytes: Vec<u8> = final_simd_ids.iter().map(|&id| id as u8).collect();

        // 2. Безопасно восстанавливаем слова, которые РЕАЛЬНО нарезал SIMD
        let mut simd_sliced_words: Vec<String> = Vec::new();
        let mut current_word = Vec::new();

        // Просто идем по байтам, которые выдал SIMD.
        // Если в буфере SIMD есть пробелы (32, 10, 9), значит сплиттер их НЕ отфильтровал!
        for &b in &simd_bytes {
            let is_space = b == 32 || b == 10 || b == 9 || b == 13;
            if !is_space {
                current_word.push(b);
            } else if !current_word.is_empty() {
                let word_str = String::from_utf8_lossy(&current_word).into_owned();
                simd_sliced_words.push(word_str);
                current_word.clear();
                // Запишем сам факт того, что SIMD пропустил мусорный символ
                simd_sliced_words.push(format!("[ОШИБКА: Сплиттер пропустил байт {}]", b));
            }
        }
        if !current_word.is_empty() {
            let word_str = String::from_utf8_lossy(&current_word).into_owned();
            simd_sliced_words.push(word_str);
        }

        println!("--- РЕЗУЛЬТАТ СЛИЧЕНИЯ СЛОВ ИЗ SIMD ---");

        // Выведем то, что реально получилось в SIMD
        for (idx, w) in simd_sliced_words.iter().enumerate() {
            println!("SIMD Word №{}: '{}'", idx, w);
        }

        // Проверяем, что мусора в списке нет
        let has_errors = simd_sliced_words.iter().any(|w| w.contains("ОШИБКА"));
        assert!(!has_errors, "Тест провален! SIMD-сплиттер пропускает пробельные символы в буфер токенов!");

        // Проверка порядка и наличия
        let mut last_found_index = 0;
        for actual_word in &simd_sliced_words {
            assert!(
                russian_prompt.contains(actual_word),
                "Слова '{}' нет в оригинальном тексте!", actual_word
            );
            let current_word_index = russian_prompt[last_found_index..]
                .find(actual_word)
                .map(|idx| idx + last_found_index)
                .unwrap();
            last_found_index = current_word_index + actual_word.len();
        }
    }
}
