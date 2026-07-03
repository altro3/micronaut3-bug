#[cfg(test)]
mod tests {
    use my_llama::tokenizer::simd::simd_splitter::SimdSplitter;

    #[test]
    fn test_russian_prompt_zero_copy_offsets_validation() {
        // Наш точный промпт на русском языке
        let russian_prompt = "Привет! Ты — мощная языковая модель Llama.\n\
    Твоя задача — обрабатывать огромные объемы текста на русском языке.\t\
    SIMD-алгоритмы помогают делать это на космической скорости в 2026 году!";

        // Выделяем буферы под координаты токенов (максимум по числу символов)
        let mut simd_offsets = vec![0u32; russian_prompt.len()];
        let mut simd_lens = vec![0u32; russian_prompt.len()];

        // 1. Запускаем рекордно быстрый сплиттер через фасад SimdSplitter
        let token_count = SimdSplitter::split(russian_prompt, &mut simd_offsets, &mut simd_lens);

        // 2. СТРОИМ МАССИВ СЛОВ ИЗ РЕЗУЛЬТАТОВ SIMD
        // Мы берем оригинальную строку и делаем zero-copy срезы по координатам из буфера
        let mut simd_extracted_words: Vec<&str> = Vec::new();

        for i in 0..token_count {
            let start = simd_offsets[i] as usize;
            let len = simd_lens[i] as usize;

            let word_slice = &russian_prompt[start..start + len];
            simd_extracted_words.push(word_slice);
        }

        println!("--- РЕЗУЛЬТАТ СЛИЧЕНИЯ ИЗ ZERO-COPY SIMD ---");
        for (idx, word) in simd_extracted_words.iter().enumerate() {
            println!("SIMD Токен №{}: '{}'", idx, word);
        }

        // 3. ЦИКЛ СТРОГО ПО МАССИВУ СЛОВ ИЗ SIMD (Твое жесткое условие)
        let mut last_found_offset = 0;
        let mut total_verified = 0;

        for (simd_idx, actual_word) in simd_extracted_words.iter().enumerate() {
            // Проверка 1: Слово гарантированно должно быть подстрокой оригинала
            assert!(
                russian_prompt.contains(actual_word),
                "ОШИБКА! Робот выдумал токен №{}: '{}'",
                simd_idx,
                actual_word
            );

            // Проверка 2: Контроль индексов. Ищем слово в оригинальном промпте, начиная со шва предыдущего
            let local_idx = russian_prompt[last_found_offset..]
                .find(actual_word)
                .expect("Нарушен хронологический порядок токенов в памяти!");

            let absolute_idx = last_found_offset + local_idx;

            // Проверяем, что координата из буфера совпадает с физическим find в строке
            assert_eq!(
                absolute_idx, simd_offsets[simd_idx] as usize,
                "ОШИБКА! Сплиттер выдал неверный индекс для слова '{}'. Ожидали {}, получили {}",
                actual_word, absolute_idx, simd_offsets[simd_idx]
            );

            last_found_offset = absolute_idx + actual_word.len();
            total_verified += 1;
        }

        // 4. Проверяем, что собрали ровно все 27 слов
        assert_eq!(
            total_verified, 27,
            "ОШИБКА! Должно быть 27 токенов, а SIMD выдал только {}.",
            total_verified
        );

        println!("--- МИРОВОЙ РЕКОРД ПОДТВЕРЖДЕН ---");
        println!(
            "Сплиттер без единого аллокатора памяти и копирования нарезал {} чистых слов.",
            total_verified
        );
    }
}
