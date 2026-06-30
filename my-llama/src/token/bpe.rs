use crate::utils::PinnedHostBuffer;
use serde_json::{from_reader, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};

pub struct BpeTokenizer {
    encoder: HashMap<Vec<u8>, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    ranks: HashMap<Vec<u8>, u32>,
    pub eos_token_id: u32,
    byte_fallback: [u32; 256],
}

impl BpeTokenizer {
    pub fn from_file(file_path: &str) -> std::io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let json_data: Value =
            from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut ranks = HashMap::new();
        let mut byte_fallback = [0u32; 256];

        let vocab = json_data["model"]["vocab"].as_object().ok_or_else(|| {
            Error::new(
                ErrorKind::NotFound,
                "Не найден блок 'model.vocab' в tokenizer.json",
            )
        })?;

        let mut max_id = 0;

        for (token_str, id_val) in vocab {
            let id = id_val
                .as_u64()
                .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Неверный формат ID токена"))?
                as u32;

            if id > max_id {
                max_id = id;
            }

            let token_bytes: Vec<u8> = token_str.chars().map(|c| c as u8).collect();

            if token_bytes.len() == 1 {
                byte_fallback[token_bytes[0] as usize] = id;
            }

            encoder.insert(token_bytes.clone(), id);
            decoder.insert(id, token_bytes.clone());
            ranks.insert(token_bytes, id);
        }

        for b in 0..=255 {
            if byte_fallback[b] == 0 {
                byte_fallback[b] = match encoder.get(&vec![b as u8]) {
                    Some(&id) => id,
                    None => b as u32,
                };
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

        println!(
            "[УСПЕШНО] Загружен оригинальный словарь Qwen. Всего токенов: {}, EOS ID: {}",
            encoder.len(),
            eos_id
        );

        Ok(BpeTokenizer {
            encoder,
            decoder,
            ranks,
            eos_token_id: eos_id,
            byte_fallback,
        })
    }

    pub fn encode_to_pinned(&self, text: &str, pinned_dst: &mut PinnedHostBuffer) -> usize {
        let slice = pinned_dst.as_slice_mut();
        if text.is_empty() {
            if !slice.is_empty() {
                slice[0] = self.eos_token_id as f32;
                return 1;
            }
            return 0;
        }

        let raw_bytes = text.as_bytes();

        let mut byte_parts = [0usize; 4096];
        let mut parts_len = 0;

        for i in 0..=raw_bytes.len() {
            if parts_len >= byte_parts.len() {
                break;
            }
            byte_parts[parts_len] = i;
            parts_len += 1;
        }

        loop {
            if parts_len <= 2 {
                break;
            }

            let mut min_rank = u32::MAX;
            let mut best_pair_idx = None;

            for i in 0..(parts_len - 2) {
                let start = byte_parts[i];
                let end = byte_parts[i + 2];
                let slice_bytes = &raw_bytes[start..end];

                if let Some(&rank) = self.ranks.get(slice_bytes) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_pair_idx {
                for j in (idx + 1)..(parts_len - 1) {
                    byte_parts[j] = byte_parts[j + 1];
                }
                parts_len -= 1;
            } else {
                break;
            }
        }

        let mut token_count = 0;
        for i in 0..(parts_len - 1) {
            if token_count >= slice.len() {
                eprintln!("Критическая ошибка: Pinned буфер хоста слишком мал!");
                break;
            }

            let start = byte_parts[i];
            let end = byte_parts[i + 1];
            let token_bytes = &raw_bytes[start..end];

            let token_id = if let Some(&id) = self.encoder.get(token_bytes) {
                id
            } else if token_bytes.len() == 1 {
                self.byte_fallback[token_bytes[0] as usize]
            } else {
                token_bytes[0] as u32
            };

            slice[token_count] = token_id as f32;
            token_count += 1;
        }

        token_count
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut dummy_buffer = PinnedHostBuffer::new(text.len() + 2);
        let count = self.encode_to_pinned(text, &mut dummy_buffer);

        let slice = dummy_buffer.as_slice_mut();
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            result.push(slice[i] as u32);
        }
        result
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut byte_buffer = Vec::new();

        for &id in ids {
            if id == self.eos_token_id {
                continue;
            }
            if let Some(bytes) = self.decoder.get(&id) {
                byte_buffer.extend_from_slice(bytes);
            } else {
                for (b, &fallback_id) in self.byte_fallback.iter().enumerate() {
                    if fallback_id == id {
                        byte_buffer.push(b as u8);
                        break;
                    }
                }
            }
        }

        String::from_utf8_lossy(&byte_buffer).into_owned()
    }

    /// Для тестов
    pub fn new_micro() -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut ranks = HashMap::new();
        let mut byte_fallback = [0u32; 256];

        for b in 0..=255 {
            let bytes = vec![b];
            let id = b as u32;
            encoder.insert(bytes.clone(), id);
            decoder.insert(id, bytes.clone());
            ranks.insert(bytes, id);
            byte_fallback[b as usize] = id;
        }

        let eos_id = 256;
        let eos_bytes = b"<|endoftext| decay>".to_vec();
        encoder.insert(eos_bytes.clone(), eos_id);
        decoder.insert(eos_id, eos_bytes.clone());
        ranks.insert(eos_bytes, eos_id);

        let mut add_bpe_merge = |bytes: Vec<u8>, id: u32| {
            encoder.insert(bytes.clone(), id);
            decoder.insert(id, bytes.clone());
            ranks.insert(bytes, id);
        };

        add_bpe_merge(vec![208, 191], 257); // 'п'
        add_bpe_merge(vec![209, 128], 258); // 'р'
        add_bpe_merge(vec![208, 191, 209, 128], 259); // 'пр'

        BpeTokenizer {
            encoder,
            decoder,
            ranks,
            eos_token_id: eos_id,
            byte_fallback,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_flat_bpe_math() {
        let tokenizer = BpeTokenizer::new_micro();
        let text = "привет";

        let ids = tokenizer.encode(text);
        assert!(!ids.is_empty());
        assert_eq!(*ids.last().unwrap(), tokenizer.eos_token_id);

        let decoded = tokenizer.decode(&ids);
        assert!(decoded.contains("привет"));
    }

    #[test]
    fn test_tokenizer_pinned_dma_output() {
        let tokenizer = BpeTokenizer::new_micro();
        let mut pinned_buf = PinnedHostBuffer::new(32);

        let count = tokenizer.encode_to_pinned("привет", &mut pinned_buf);
        assert!(count > 0);

        let slice = pinned_buf.as_slice_mut();
        assert_eq!(slice[count - 1], tokenizer.eos_token_id as f32);
        println!(
            "[ЮНИТ-ТЕСТ УСПЕШЕН] Токенизатор успешно записал {} токенов напрямую в Pinned Memory.",
            count
        );
    }
}
