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
    pub trie_nodes: Vec<FlatTrieNode>,
    pub trie_root_offsets: [u32; 256],
}

#[derive(Copy, Clone, Debug)]
#[repr(C, align(8))]
pub struct FlatTrieNode {
    // Если этот узел — конец валидного токена, тут лежит его ID. Если нет — u32::MAX
    pub token_id: u32,
    // Смещение в общем массиве узлов `trie_nodes`, где лежат 256 дочерних переходов для этого узла.
    // Если детей нет (лист дерева) — u32::MAX
    pub children_offset: u32,
}
