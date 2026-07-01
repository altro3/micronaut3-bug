use super::bpe::BpeTokenizer;
use super::bpe_types::BpeValue;
use rustc_hash::FxHashMap;
use serde_json::{from_reader, Value};
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};

pub struct TokenizerFactory;

impl TokenizerFactory {
    fn byte_to_unicode_decode(s: &str) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(s.len());
        for c in s.chars() {
            let u = c as u32;
            let b = match u {
                0x0000..=0x0020 => u as u8,
                0x007F..=0x00A0 => u as u8,
                0x0100..=0x011F => (u - 0x0100) as u8 + 33,
                0x0120..=0x017F => (u - 0x0120) as u8 + 127,
                0x0180..=0x01A3 => (u - 0x0180) as u8 + 223,
                _ => u as u8,
            };
            bytes.push(b);
        }
        bytes
    }

    pub fn from_file(file_path: &str) -> std::io::Result<BpeTokenizer> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let json_data: Value = from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        let mut pair_ranks = FxHashMap::default();
        let mut byte_fallback = [0u32; 256];

        let vocab = json_data["model"]["vocab"]
            .as_object()
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "Vocab not found"))?;

        let mut bytes_to_id: FxHashMap<Vec<u8>, u32> = FxHashMap::default();

        for (token_str, id_val) in vocab {
            let id = id_val.as_u64().ok_or_else(|| Error::new(ErrorKind::InvalidData, "ID error"))? as u32;
            let token_bytes = Self::byte_to_unicode_decode(token_str);

            if token_bytes.len() == 1 {
                byte_fallback[token_bytes[0] as usize] = id;
            }
            bytes_to_id.insert(token_bytes, id);
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
                        let p1 = Self::byte_to_unicode_decode(parts[0]);
                        let p2 = Self::byte_to_unicode_decode(parts[1]);
                        merges.push((p1, p2));
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
                if let Some(&id) = bytes_to_id.get(p1) {
                    id1 = id;
                }
            }
            if id2 == u32::MAX {
                if let Some(&id) = bytes_to_id.get(p2) {
                    id2 = id;
                }
            }
            if id1 != u32::MAX && id2 != u32::MAX {
                let pack = ((id1 as u64) << 32) | (id2 as u64);

                let mut merged_bytes = p1.clone();
                merged_bytes.extend_from_slice(p2);

                let merged_id = bytes_to_id.get(&merged_bytes).copied().unwrap_or(rank as u32);

                pair_ranks.insert(
                    pack,
                    BpeValue {
                        rank: rank as u32,
                        id: merged_id,
                    },
                );
            }
        }

        let mut eos_id = 248044;
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
