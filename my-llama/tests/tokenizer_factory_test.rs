#[cfg(test)]
mod real_data_validation_tests {
    use my_llama::tokenizer::factory::TokenizerFactory;

    #[test]
    fn assert_qwen_model_parallel_encoding_integrity() {
        // Путь к реальному файлу конфигурации Qwen
        let real_model_path = "data/qwen_model.json";

        assert!(
            std::path::Path::new(real_model_path).exists(),
            "Тест не запущен: положите реальный tokenizer.json по пути data/qwen_model.json"
        );

        println!("\n[Тест] Запускаю парсинг реального JSON на сырых указателях...");
        let tokenizer = TokenizerFactory::from_file(real_model_path).expect("Фабрика упала при чтении реального JSON-файла!");

        // Тестовый многоязычный батч (String-строки строго под твой encode_parallel)
        let test_batch = vec![
            "The quick brown fox jumps over the lazy dog. Performance and ultra efficiency combined.".to_string(),
            "Rust performance тест без компромиссов на процессорах Core i9.".to_string(),
            "人工智能 🧠 计算 ∞ без_компромиссов_5090_∑_".to_string(),
            "Ġas ect ke rom con ĠW ĠE Ġcom Ġreturn art ĠH ack import ublic Ġor est".to_string(), // Твои токены словаря
        ];

        let original_bytes: usize = test_batch.iter().map(|s| s.len()).sum();
        println!("[Тест] Всего байт на входе: {}", original_bytes);

        // Вызываем твой параллельный энкодер на пуле потоков
        println!("[Тест] Запускаю параллельное кодирование через encode_parallel...");
        let encoded_results = tokenizer.encode_parallel(&test_batch);

        // Считаем полученные токены
        let total_tokens: usize = encoded_results.iter().map(|v| v.len()).sum();
        println!("[Тест] Всего токенов на выходе: {}", total_tokens);

        let compression_ratio = original_bytes as f64 / total_tokens as f64;
        println!("[Тест] Полученный коэффициент BPE-сжатия: {:.2}x", compression_ratio);

        // --- ЖЕСТКИЕ АССЕРТЫ ---

        // 1. Проверяем, что результаты вернулись для ВСЕХ строк батча
        assert_eq!(
            encoded_results.len(),
            test_batch.len(),
            "Критический сбой: Количество выходных векторов не совпадает с батчем!"
        );

        // 2. Главный ассерт: теперь, когда byte_fallback защищен от перезаписи,
        // ID совпали, граф мёрджей включился, и сжатие обязано улететь далеко вверх (обычно 2.5x - 3.5x).
        assert!(
            compression_ratio > 1.4,
            "КРИТИЧЕСКИЙ БАГ: Коэффициент BPE-сжатия равен {:.2}x! Токенизатор не склеивает пары.",
            compression_ratio
        );

        println!("[Тест] УСПЕХ: Настоящая модель Qwen успешно скомпилирована фабрикой!");
        println!("[Тест] Параллельный рантайм выдал честное BPE-сжатие без паник и сегфолтов.");
    }

    #[test]
    fn assert_qwen_model_parallel_encoding_integrity_english() {
        let real_model_path = "data/qwen_model.json";

        assert!(
            std::path::Path::new(real_model_path).exists(),
            "Положите реальный tokenizer.json по пути data/qwen_model.json"
        );

        println!("\n[Тест] Запускаю парсинг реального JSON на сырых указателях...");
        let tokenizer = TokenizerFactory::from_file(real_model_path)
            .expect("Фабрика упала при чтении реального JSON-файла!");

        // ИСКЛЮЧИТЕЛЬНО ЛАТИНСКИЙ ТЕКСТ (Эталонные частые BPE-конструкции)
        let test_batch = vec![
            "the quick brown fox jumps over the lazy dog".to_string(),
            "performance and ultra efficiency combined with Rust performance text without compromise".to_string(),
            "function const void System set return art hack import public or est".to_string(),
            "but function const void System set ep ally validation successful tokens found".to_string(),
        ];

        let original_bytes: usize = test_batch.iter().map(|s| s.len()).sum();
        println!("[Тест] Всего байт на входе (Латиница): {}", original_bytes);

        // Вызываем твой параллельный энкодер на пуле потоков
        println!("[Тест] Запускаю параллельное кодирование через encode_parallel...");
        let encoded_results = tokenizer.encode_parallel(&test_batch);

        // Считаем полученные токены
        let total_tokens: usize = encoded_results.iter().map(|v| v.len()).sum();
        println!("[Тест] Всего токенов на выходе: {}", total_tokens);

        let compression_ratio = original_bytes as f64 / total_tokens as f64;
        println!("[Тест] Полученный коэффициент BPE-сжатия на латинице: {:.2}x", compression_ratio);

        // Ставим честный, жесткий ассерт на сжатие латиницы
        assert!(
            compression_ratio > 2.0,
            "КРИТИЧЕСКИЙ БАГ: Сжатие латиницы всего {:.2}x! Либо в encode_single_chunk не хватает Byte-Level маппинга пробелов, либо граф мёрджей не сошёлся.",
            compression_ratio
        );

        println!("[Тест] УСПЕХ: Настоящая модель Qwen на латинице выдала честное BPE-сжатие!");
    }

    #[test]
    fn debug_byte_fallback_vs_merges() {
        let real_model_path = "data/qwen_model.json";
        let tokenizer = TokenizerFactory::from_file(real_model_path).unwrap();

        // Печатаем ID для букв 'i' и 'n' из нашего byte_fallback
        // В логе мёрджей мы видели, что мёрдж ожидает: 'i' = 72, 'n' = 77
        let id_i_fallback = tokenizer.byte_fallback[b'i' as usize];
        let id_n_fallback = tokenizer.byte_fallback[b'n' as usize];
        let id_space_fallback = tokenizer.byte_fallback[0x20]; // пробел 'Ġ' -> 32

        println!("\n[КОНТРОЛЬ ID БАЙТ]");
        println!("|-> В byte_fallback для 'i' записан ID: {} (Ожидалось: 72)", id_i_fallback);
        println!("|-> В byte_fallback для 'n' записан ID: {} (Ожидалось: 77)", id_n_fallback);
        println!("|-> В byte_fallback для пробела (0x20) записан ID: {}", id_space_fallback);

        println!("--- КОНЕЦ ПРОВЕРКИ fallback ---");
    }
}
