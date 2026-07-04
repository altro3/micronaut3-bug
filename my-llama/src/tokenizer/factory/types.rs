use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

#[derive(Serialize, Deserialize)]
pub struct BpeModelFields {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<[String; 2]>,
}

#[derive(Serialize, Deserialize)]
pub struct RegexPattern {
    #[serde(rename = "Regex")]
    pub regex: String,
}

#[derive(Serialize, Deserialize)]
pub struct PreTokenizerEntry {
    pub pattern: Option<RegexPattern>,
}

#[derive(Serialize, Deserialize)]
pub struct PreTokenizerFields {
    pub pretokenizers: Vec<PreTokenizerEntry>,
}

#[derive(Serialize, Deserialize)]
pub struct QwenJsonModel {
    pub version: String,
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
    pub trie_nodes: Vec<FlatTrieNode>,
    pub trie_root_offsets: [u32; 256],
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(8))]
pub struct FlatTrieNode {
    pub token_id: u32,
    pub children_offset: u32,
}
