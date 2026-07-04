use super::decoder::HfByteDecoder;
use super::types::{CompiledVocabulary, QwenJsonModel};
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Result};

pub struct DictCompiler;

impl DictCompiler {
    pub fn compile_from_json(file_path: &str) -> Result<CompiledVocabulary> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let root: QwenJsonModel = serde_json::from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        let eos_token_id = Self::extract_eos(&root);
        let vocab_size = root.model.vocab.len();

        let mut byte_fallback = [u32::MAX; 256];
        let mut vocab_compiled_tokens = vec![Vec::new(); vocab_size.max(260000)];
        for (token_str, &id) in root.model.vocab.iter() {
            let raw_bytes = HfByteDecoder::decode_string(token_str);
            let id_idx = id as usize;

            if id_idx >= vocab_compiled_tokens.len() {
                vocab_compiled_tokens.resize(id_idx + 1, Vec::new());
            }
            vocab_compiled_tokens[id_idx] = raw_bytes.clone();

            if raw_bytes.len() == 1 {
                byte_fallback[raw_bytes[0] as usize] = id;
            }
        }

        Self::fill_qwen_byte_fallbacks(&root, &mut byte_fallback);

        let mut raw_pairs = Vec::with_capacity(root.model.merges.len());
        for (rank, pair) in root.model.merges.iter().enumerate() {
            let left_str = &pair[0];
            let right_str = &pair[1];

            if let (Some(&id1), Some(&id2)) = (root.model.vocab.get(left_str), root.model.vocab.get(right_str)) {
                let mut merged_str = String::with_capacity(left_str.len() + right_str.len());
                merged_str.push_str(left_str);
                merged_str.push_str(right_str);

                if let Some(&target_id) = root.model.vocab.get(&merged_str) {
                    let packed_key = ((id1 as u64) << 32) | (id2 as u64);
                    raw_pairs.push((packed_key, (rank as u32, target_id)));
                }
            }
        }

        Ok(CompiledVocabulary {
            byte_fallback,
            raw_pairs,
            eos_token_id,
            vocab_size,
            vocab_compiled_tokens,
        })
    }

    fn extract_eos(root: &QwenJsonModel) -> u32 {
        if let Some(ref added) = root.added_tokens {
            for tok in added {
                if tok.content == "<|endoftext|>" {
                    return tok.id;
                }
            }
        }
        248044
    }

    fn fill_qwen_byte_fallbacks(root: &QwenJsonModel, fallback: &mut [u32; 256]) {
        for b in 0..=255 {
            if fallback[b] == u32::MAX {
                let byte_token_name = format!("<|byte_{:02X}|>", b);
                if let Some(&id) = root.model.vocab.get(&byte_token_name) {
                    fallback[b] = id;
                } else {
                    fallback[b] = (root.model.vocab.len() as u32) + (b as u32);
                }
            }
        }
    }
}
