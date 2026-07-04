// Файл: tests/tokenizer/tokenizer_integration_tests.rs

#[cfg(test)]
mod integration_tests {
    use my_llama::cuda::PinnedHostBuffer;
    use my_llama::tokenizer::bpe::context::TokenizationContext;
    use my_llama::tokenizer::bpe::pipeline::TokenizerPipeline;
    use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
    use my_llama::tokenizer::BpeTokenizer;

    fn create_mock_tokenizer() -> BpeTokenizer {
        let mut raw_pairs = Vec::new();
        let mut byte_fallback = [0u32; 256];
        let mut vocab_compiled_tokens = vec![Vec::new(); 512];

        for b in 0..=255 {
            byte_fallback[b] = b as u32;
            vocab_compiled_tokens[b] = vec![b as u8];
        }

        raw_pairs.push((((104u64) << 32) | 101u64, (1, 256)));
        vocab_compiled_tokens[256] = b"he".to_vec();

        raw_pairs.push((((108u64) << 32) | 108u64, (2, 257)));
        vocab_compiled_tokens[257] = b"ll".to_vec();

        raw_pairs.push((((111u64) << 32) | 44u64, (3, 258)));
        vocab_compiled_tokens[258] = b"o,".to_vec();

        raw_pairs.push((((256u64) << 32) | 257u64, (0, 259)));
        vocab_compiled_tokens[259] = b"hell".to_vec();

        BpeTokenizer::new(&raw_pairs, byte_fallback, 50256, 512, &vocab_compiled_tokens)
    }

    fn create_mock_dfa() -> FlatDfaRuntime {
        let trans_bytes = vec![0u8; 1 * 256 * 2];
        let accept_bytes = vec![1u8; 1];
        FlatDfaRuntime::from_binary_dump(1, &trans_bytes, &accept_bytes)
    }

    fn create_pipeline() -> TokenizerPipeline {
        TokenizerPipeline::new(create_mock_tokenizer(), create_mock_dfa())
    }

    #[test]
    fn test_single_thread_and_short_path_correctness() {
        let pipeline = create_pipeline();
        let mut ctx = TokenizationContext::new(512, 1024);
        let tokens = pipeline.encode("hello,world", &mut ctx);

        assert_eq!(tokens[0], 259);
        assert_eq!(tokens[1], 258);
        assert_eq!(tokens[2], 119); // 'w'
    }

    #[test]
    fn test_encode_to_pinned_cuda_buffer() {
        let pipeline = create_pipeline();
        let mut ctx = TokenizationContext::new(512, 1024);
        let mut pinned_buffer = PinnedHostBuffer::new(32);

        let copied = pipeline.encode_to_pinned("hello,world", &mut pinned_buffer, &mut ctx);
        assert!(copied > 0);

        let slice = pinned_buffer.as_slice_mut();
        assert_eq!(slice[0], 259.0f32);
        assert_eq!(slice[1], 258.0f32);
    }

    #[test]
    fn test_encode_parallel_stress_and_safety() {
        let pipeline = create_pipeline();
        let mut texts = Vec::with_capacity(1000);
        for i in 0..1000 {
            if i % 2 == 0 { texts.push("hello,world".to_string()); } else { texts.push("hello,hello,hello".to_string()); }
        }

        let mut contexts: Vec<TokenizationContext> = std::iter::repeat_with(|| TokenizationContext::new(512, 1024))
            .take(8)
            .collect();

        let batch_results = pipeline.encode_parallel(&texts, &mut contexts);
        assert_eq!(batch_results.len(), 1000);
        assert_eq!(batch_results[0][0], 259);
    }
}

#[cfg(test)]
mod real_data_validation_tests {
    use my_llama::tokenizer::factory::compiler::DictCompiler;
    use my_llama::tokenizer::bpe::pipeline::TokenizerPipeline;
    use my_llama::tokenizer::bpe::context::TokenizationContext;
    use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
    use my_llama::tokenizer::BpeTokenizer;

    fn create_mock_dfa() -> FlatDfaRuntime {
        let trans_bytes = vec![0u8; 1 * 256 * 2];
        let accept_bytes = vec![1u8; 1];
        FlatDfaRuntime::from_binary_dump(1, &trans_bytes, &accept_bytes)
    }

    #[test]
    fn assert_qwen_model_parallel_encoding_integrity() {
        let real_model_path = "data/qwen_model.json";
        if !std::path::Path::new(real_model_path).exists() { return; } // Пропускаем, если нет файла данных

        let compiled_vocab = DictCompiler::compile_from_json(real_model_path).unwrap();
        let bpe_tokenizer = BpeTokenizer::new(
            &compiled_vocab.raw_pairs, compiled_vocab.byte_fallback,
            compiled_vocab.eos_token_id, compiled_vocab.vocab_size, &compiled_vocab.vocab_compiled_tokens,
        );

        let pipeline = TokenizerPipeline::new(bpe_tokenizer, create_mock_dfa());
        let test_batch = vec!["The memory management subsystem is complex.".to_string()];

        let mut contexts: Vec<TokenizationContext> = std::iter::repeat_with(|| TokenizationContext::new(compiled_vocab.vocab_size, 1024))
            .take(1)
            .collect();

        let encoded_results = pipeline.encode_parallel(&test_batch, &mut contexts);
        assert!(!encoded_results.is_empty());
    }
}
