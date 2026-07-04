use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
}

#[derive(Deserialize)]
pub struct BpeModelFields {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<[String; 2]>,
}

#[derive(Deserialize)]
pub struct RegexPattern {
    #[serde(rename = "Regex")]
    pub regex: String,
}

#[derive(Deserialize)]
pub struct PreTokenizerEntry {
    pub pattern: Option<RegexPattern>,
}

#[derive(Deserialize)]
pub struct PreTokenizerFields {
    pub pretokenizers: Vec<PreTokenizerEntry>,
}

#[derive(Deserialize)]
pub struct QwenJsonModel {
    pub added_tokens: Option<Vec<AddedToken>>,
    pub pre_tokenizer: PreTokenizerFields,
    pub model: BpeModelFields,
}

pub struct CompiledVocabulary {
    pub byte_fallback: [u32; 256],
    pub raw_pairs: Vec<(u64, (u32, u32))>,
    pub eos_token_id: u32,
    pub vocab_size: usize,
    pub vocab_compiled_tokens: Vec<Vec<u8>>,
    pub extracted_regex: String,
}
