use super::bpe::BpeTokenizer;
use super::bpe_types::BpeValue;
use rustc_hash::FxHashMap;
use serde_json::{from_reader, Value};
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};

pub struct TokenizerFactory;

impl TokenizerFactory {
    pub fn from_file(file_path: &str) -> std::io::Result<BpeTokenizer> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let json_data: Value = from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        let mut pair_ranks = FxHashMap::default();
        let mut byte_fallback = [0u32; 256];

        let vocab = json_data["model"]["vocab"]
            .as_object()
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "Vocab not found"))?;

        for (token_str, id_val) in vocab {
            let id = id_val.as_u64().ok_or_else(|| Error::new(ErrorKind::InvalidData, "ID error"))? as u32;
            let token_bytes = token_str.as_bytes().to_vec();

            if token_bytes.len() == 1 {
                byte_fallback[token_bytes[0] as usize] = id;
            }
        }

        for b in 0..=255 {
            if byte_fallback[b] == 0 {
                byte_fallback[b] = b as u32;
            }
        }

        let mut merges = Vec::new();
        if let Some(merges_json) = json_data["model"]["merges"].as_array() {
            for item in merges_json {
                if let Some(merge_str) = item.as_str() {
                    let parts: Vec<&str> = merge_str.split(' ').collect();
                    if parts.len() == 2 {
                        merges.push((parts[0].as_bytes().to_vec(), parts[1].as_bytes().to_vec()));
                    }
                }
            }
        }

        for (rank, (p1, p2)) in merges.iter().enumerate() {
            let mut id1 = u32::MAX;
            let mut id2 = u32::MAX;

            if p1.len() == 1 {
                id1 = byte_fallback[p1[0] as usize];
            }
            if p2.len() == 1 {
                id2 = byte_fallback[p2[0] as usize];
            }

            if id1 == u32::MAX {
                if let Some(id) = vocab.get(&String::from_utf8_lossy(p1).into_owned()).and_then(|v| v.as_u64()) {
                    id1 = id as u32;
                }
            }
            if id2 == u32::MAX {
                if let Some(id) = vocab.get(&String::from_utf8_lossy(p2).into_owned()).and_then(|v| v.as_u64()) {
                    id2 = id as u32;
                }
            }

            if id1 != u32::MAX && id2 != u32::MAX {
                let pack = ((id1 as u64) << 32) | (id2 as u64);

                let mut merged_bytes = p1.clone();
                merged_bytes.extend_from_slice(p2);
                let merged_str = String::from_utf8_lossy(&merged_bytes).into_owned();

                let merged_id = vocab.get(&merged_str).and_then(|v| v.as_u64()).map(|id| id as u32).unwrap_or(rank as u32);

                pair_ranks.insert(
                    pack,
                    BpeValue {
                        rank: rank as u32,
                        id: merged_id,
                    },
                );
            }
        }

        let mut eos_id = 151643;
        if let Some(added_tokens) = json_data["added_tokens"].as_array() {
            for token in added_tokens {
                if token["content"].as_str() == Some("<|endoftext|>") {
                    if let Some(id) = token["id"].as_u64() {
                        eos_id = id as u32;
                    }
                }
            }
        }

        Ok(BpeTokenizer::new(pair_ranks, byte_fallback, eos_id))
    }
}
