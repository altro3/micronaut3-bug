#[cfg(test)]
mod integration_tests {
    use my_llama::cuda::PinnedHostBuffer;
    use my_llama::tokenizer::context::TokenizationContext;
    use my_llama::tokenizer::BpeTokenizer;

    // Вспомогательная функция для генерации фейкового, но валидного словаря BPE
    fn create_mock_tokenizer() -> BpeTokenizer {
        let mut raw_pairs = Vec::new();
        let mut byte_fallback = [0u32; 256];

        // 1. Инициализируем byte_fallback: пусть байты 0..255 маппятся в ID токенов 0..255
        for b in 0..=255 {
            byte_fallback[b] = b as u32;
        }

        // 2. Создаем правила слияния (rules).
        // Формат ключа pack: ((left_id as u64) << 32) | (right_id as u64)
        // Правило 1: 'h' (104) + 'e' (101) -> Токен ID 256, Ранг 1 (высокий приоритет)
        let pack_he = ((104u64) << 32) | 101u64;
        raw_pairs.push((pack_he, (1, 256)));

        // Правило 2: 'l' (108) + 'l' (108) -> Токен ID 257, Ранг 2
        let pack_ll = ((108u64) << 32) | 108u64;
        raw_pairs.push((pack_ll, (2, 257)));

        // Правило 3: 'o' (111) + ',' (44) -> Токен ID 258, Ранг 3
        let pack_o_comma = ((111u64) << 32) | 44u64;
        raw_pairs.push((pack_o_comma, (3, 258)));

        // Правило 4: Токен 256 ("he") + токен 257 ("ll") -> Токен ID 259 ("hell"), Ранг 0 (наивысший приоритет!)
        let pack_hell = ((256u64) << 32) | 257u64;
        raw_pairs.push((pack_hell, (0, 259)));

        BpeTokenizer::new(&raw_pairs, byte_fallback, 50256, 512)
    }

    #[test]
    fn test_single_thread_and_short_path_correctness() {
        let tokenizer = create_mock_tokenizer();
        let mut ctx = TokenizationContext::new(512, 1024);

        // Входной текст. "hello," содержит "he" -> 256, "ll" -> 257. Затем они должны слиться в 259 ("hell").
        // 'o' и ',' сольются в 258. Останется 'w', 'o', 'r', 'l', 'd'.
        let text = "hello,world";

        // Запускаем наш оптимизированный однопоточный энкодер
        let tokens = tokenizer.encode(text, &mut ctx);

        println!("Финальные токены: {:?}", tokens);

        // Ожидаемый результат слияний:
        // "he" + "ll" -> 259
        // "o" + "," -> 258
        // 'w' (119), 'o' (111), 'r' (114), 'l' (108), 'd' (100)
        assert_eq!(tokens[0], 259, "Ошибка BPE: 'hell' не собрался!");
        assert_eq!(tokens[1], 258, "Ошибка BPE: 'o,' не собрался!");
        assert_eq!(tokens[2], 119); // 'w'
    }

    #[test]
    fn test_encode_to_pinned_cuda_buffer() {
        let tokenizer = create_mock_tokenizer();
        let mut ctx = TokenizationContext::new(512, 1024);
        let mut pinned_buffer = PinnedHostBuffer::new(32);

        let text = "hello,world";
        let copied = tokenizer.encode_to_pinned(text, &mut pinned_buffer, &mut ctx);

        assert!(copied > 0, "Токены не были скопированы в Pinned память");

        let slice = pinned_buffer.as_slice_mut();
        // Проверяем, что наши Aligned Non-Temporal Stores (_mm256_stream_ps)
        // корректно сконвертировали u32 в f32 и уложили их в RAM
        assert_eq!(slice[0], 259.0f32, "Ошибка SIMD-выгрузки в CUDA буфер");
        assert_eq!(slice[1], 258.0f32);
        println!("CUDA Pinned буфер успешно верифицирован и готов к PCIe трансляции!");
    }

    #[test]
    fn test_encode_parallel_stress_and_safety() {
        let tokenizer = create_mock_tokenizer();

        // Генерируем массив из 1000 строк для жесткого стресс-теста пула потоков
        let mut texts = Vec::with_capacity(1000);
        for i in 0..1000 {
            if i % 2 == 0 {
                texts.push("hello,world".to_string());
            } else {
                texts.push("hello,hello,hello".to_string());
            }
        }

        // Запускаем наше многопоточное ядро, очищенное от конкуренции за аллокатор ОС
        let batch_results = tokenizer.encode_parallel(&texts);

        assert_eq!(batch_results.len(), 1000);

        // Проверяем выборочные индексы на предмет Memory Corruption / False Sharing
        for i in (0..1000).step_by(50) {
            let res = &batch_results[i];
            if i % 2 == 0 {
                assert_eq!(res[0], 259, "Параллельный поток повредил данные на индексе {}", i);
                assert_eq!(res[1], 258);
            } else {
                // "hello,hello,hello" -> "hell" (259), "o," (258), "hell" (259), "o," (258), 'h','e','l','l','o'
                assert_eq!(res[0], 259);
                assert_eq!(res[1], 258);
            }
        }
        println!("Стресс-тест параллелизации пройден: 1000 текстов обработаны zero-lock!");
    }
}
