#[cfg(test)]
mod real_data_validation_tests {
    use my_llama::tokenizer::factory::TokenizerFactory;

    #[test]
    fn assert_qwen_model_parallel_encoding_integrity() {
        let real_model_path = "data/qwen_model.json";

        assert!(
            std::path::Path::new(real_model_path).exists(),
            "Положите реальный tokenizer.json по пути data/qwen_model.json"
        );

        println!("\n[Тест] Запускаю парсинг реального JSON на сырых указателях...");
        let tokenizer = TokenizerFactory::from_file(real_model_path)
            .expect("Фабрика упала при чтении реального JSON-файла!");

        let test_batch = vec![
            "The memory management subsystem is one of the most complex parts of the operating system kernel. \
            Efficient memory allocation and high performance page tables are critical for modern computer architectures. \
            The implementation utilizes advanced algorithms to minimize synchronization overhead and context switching. \
            Virtual memory mapping guarantees isolated address spaces for concurrent execution threads without compromise. \
            High throughput data processing functions require direct hardware optimization and vectorized instruction sets. \
            The architecture designed here achieves maximum processing capability by executing parallel tasks simultaneously.".to_string(),

            "The system architecture incorporates an inline hash table design to optimize lookups in constant time. \
            Every character sequence and implementation details are carefully mapped to avoid cache line execution conflicts. \
            We implement high performance stream processing for massive network utilization and structured buffer configuration. \
            The communication layer utilizes specialized asynchronous execution techniques to achieve ultra low latency response times. \
            Hardware accelerated processing implementation combined with efficient software abstraction guarantees zero copy execution.".to_string(),

            "public unsafe fn execution_block_optimization(context: &mut SystemContext, buffer: *mut u8) -> usize { \
            let current_execution_pointer = context.instruction_pointer; \
            let memory_allocation_size = context.buffer_capacity.next_power_of_two(); \
            if memory_allocation_size > context.maximum_limit { \
                return context.error_handling_routine(SystemError::MemoryAllocationFailure); \
            } \
            std::ptr::copy_nonoverlapping(buffer, context.destination_pointer, memory_allocation_size); \
            context.synchronization_barrier(); \
            return memory_allocation_size; \
            }".to_string()
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
            "КРИТИЧЕСКИЙ БАГ: Сжатие латиницы всего {:.2}x!",
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
