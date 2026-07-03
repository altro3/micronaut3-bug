#[cfg(test)]
mod tests {
    use my_llama::tokenizer::simd::simd_splitter::SimdSplitter;

    #[test]
    fn test_multilang_prompt_strict_zero_copy_splitter() {
        // Тяжелый мультиязычный промпт со спецсимволами, табами, кодом и иероглифами.
        let complex_prompt = "Привет! Ты — мощная языковая модель Llama-3.1-v2.\n\
    \tdef train_fast_tokenizer(data: &[u8]) -> usize {\n\
    \t    let speed = \"космическая скорость\"; // SIMD-алгоритмы помогают!\n\
    \t    println!(\"Performance: {}, CPU: Arrow Lake\", speed);\n\
    \t}\n\
    Unbelievable industrial-grade artificial intelligence performance test on Rust.\n\
    === Юникод тест: 🚀🤖🔥 ===\n\
    === Азиатский тест (Иероглифы): 人工智能 / 大语言模型 / 速度 ===\n\
    \n\
    \t\t[SYSTEM OVERRIDE]: active_mode = true;\r\n  ";

        let mut simd_offsets = vec![0u32; complex_prompt.len()];
        let mut simd_lens = vec![0u32; complex_prompt.len()];

        // 1. Вызываем наш ИИ-сплиттер
        let token_count = SimdSplitter::split(complex_prompt, &mut simd_offsets, &mut simd_lens);

        // 2. Вырезаем слова и пробельные токены без копирования памяти
        let mut simd_extracted_tokens: Vec<&str> = Vec::new();
        for i in 0..token_count {
            let start = simd_offsets[i] as usize;
            let len = simd_lens[i] as usize;
            simd_extracted_tokens.push(&complex_prompt[start..start + len]);
        }

        println!("--- MULTILANG LLM SPLITTER OUTPUT (ZERO-COPY) ---");
        for (idx, token) in simd_extracted_tokens.iter().enumerate() {
            let visual_token = token
                .replace(" ", "[SPACE]")
                .replace("\n", "[NEWLINE\\n]")
                .replace("\t", "[TAB\\t]")
                .replace("\r", "[CR\\r]");
            println!("Token #{:02}: '{}'", idx, visual_token);
        }

        // 3. Контроль порядка и границ UTF-8
        let mut current_offset = 0;

        for (idx, token) in simd_extracted_tokens.iter().enumerate() {
            let token_len = token.len();

            let original_chunk = &complex_prompt[current_offset..current_offset + token_len];

            assert_eq!(
                *token, original_chunk,
                "ERROR! Token #{} '{}' is out of sync with original text. Expected '{}'",
                idx, token, original_chunk
            );

            let first_byte = token.as_bytes()[0];
            let is_whitespace_token = first_byte == 32 || first_byte == 10 || first_byte == 9 || first_byte == 13;

            for &b in token.as_bytes() {
                let current_byte_is_whitespace = b == 32 || b == 10 || b == 9 || b == 13;
                assert_eq!(
                    is_whitespace_token, current_byte_is_whitespace,
                    "ERROR! Token #{} '{}' mixed letters and spaces together!",
                    idx, token
                );
            }

            current_offset += token_len;
        }

        // 4. Финальная склейка без потерь
        let reconstructed_prompt: String = simd_extracted_tokens.concat();
        assert_eq!(
            reconstructed_prompt, complex_prompt,
            "ERROR! Data corruption! Reconstructed text does not match original prompt."
        );

        println!("--- MULTILANG LLM SPLITTER INTEGRATION TEST PASSED ---");
        println!("Total tokens generated: {}", token_count);
    }
}
