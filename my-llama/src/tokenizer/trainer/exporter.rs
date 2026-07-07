use crate::tokenizer::factory::types::{AddedToken, BpeModelFields, PreTokenizerEntry, PreTokenizerFields, QwenJsonModel, RegexPattern};
use crate::tokenizer::trainer::config::TrainerConfig;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::io::{Error, ErrorKind};

pub struct VocabularyExporter;

impl VocabularyExporter {
    pub fn export_qwen_json(
        output_json_path: &str,
        config: &TrainerConfig,
        cyrillic_regex: &str,
        vocab: HashMap<String, u32>,
        merges: Vec<[String; 2]>,
        last_token_id: u32,
    ) -> std::io::Result<()> {
        let eos_token = AddedToken {
            id: last_token_id,
            content: "<|endoftext|>".to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        };

        let pre_tokenizer_fields = PreTokenizerFields {
            pretokenizers: vec![PreTokenizerEntry { pattern: Some(RegexPattern { regex: cyrillic_regex.to_string() }) }],
        };

        let model_json = QwenJsonModel {
            version: "1.0".to_string(),
            added_tokens: Some(vec![eos_token]),
            pre_tokenizer: pre_tokenizer_fields,
            model: BpeModelFields { vocab, merges },
        };

        let mut buffer = Vec::with_capacity(config.io_buffer_size);
        sonic_rs::to_writer(&mut buffer, &model_json).map_err(|e| Error::new(ErrorKind::InvalidData, format!("Ошибка Serde JSON: {:?}", e)))?;
        let mut file = File::create(output_json_path)?;
        file.write_all(&buffer)?;

        Ok(())
    }
}
