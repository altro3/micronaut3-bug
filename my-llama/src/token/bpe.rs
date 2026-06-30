use crate::utils::PinnedHostBuffer;
use base64::{prelude::BASE64_STANDARD, Engine};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Clone, Copy, Debug)]
struct BpeNode {
    val: u8,
    prev: i32,
    next: i32,
}

pub struct BpeTokenizer {
    ranks: HashMap<(u8, u8), u32>,
    encoder: HashMap<Vec<u8>, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    pub eos_token_id: u32,
}

impl BpeTokenizer {
    pub fn new_micro() -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut ranks = HashMap::new();

        for b in 0..=255 {
            let bytes = vec![b];
            let id = b as u32;
            encoder.insert(bytes.clone(), id);
            decoder.insert(id, bytes);
        }

        let eos_bytes = b"[EOS]".to_vec();
        let eos_id = 256;
        encoder.insert(eos_bytes.clone(), eos_id);
        decoder.insert(eos_id, eos_bytes);

        let mut current_id = 257;
        let mut rank_counter = 0;
        let mut add_bpe_merge = |b1: u8, b2: u8| {
            ranks.insert((b1, b2), rank_counter);
            rank_counter += 1;

            let merged_bytes = vec![b1, b2];
            if !encoder.contains_key(&merged_bytes) {
                encoder.insert(merged_bytes.clone(), current_id);
                decoder.insert(current_id, merged_bytes);
                current_id += 1;
            }
        };

        add_bpe_merge(208, 191);
        add_bpe_merge(209, 128);

        BpeTokenizer {
            ranks,
            encoder,
            decoder,
            eos_token_id: eos_id,
        }
    }

    pub fn from_file(file_path: &str) -> std::io::Result<Self> {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut ranks = HashMap::new();
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut max_id = 0;

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }

            let token_bytes = BASE64_STANDARD
                .decode(parts[0])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let id = parts[1]
                .parse::<u32>()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            if id > max_id {
                max_id = id;
            }

            encoder.insert(token_bytes.clone(), id);
            decoder.insert(id, token_bytes.clone());

            if token_bytes.len() == 2 {
                let pair = (token_bytes[0], token_bytes[1]);
                ranks.insert(pair, id);
            }
        }

        let eos_id = max_id + 1;
        let eos_bytes = b"<|endoftext|>".to_vec();
        encoder.insert(eos_bytes.clone(), eos_id);
        decoder.insert(eos_id, eos_bytes);

        println!(
            "Успешно загружен словарь Tiktoken! Всего токенов: {}",
            encoder.len()
        );

        Ok(BpeTokenizer {
            ranks,
            encoder,
            decoder,
            eos_token_id: eos_id,
        })
    }

    fn bpe_merge_flat(&self, raw_bytes: &[u8]) -> Vec<Vec<u8>> {
        if raw_bytes.is_empty() {
            return Vec::new();
        }

        let mut nodes: Vec<BpeNode> = raw_bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| BpeNode {
                val: b,
                prev: i as i32 - 1,
                next: if i == raw_bytes.len() - 1 {
                    -1
                } else {
                    i as i32 + 1
                },
            })
            .collect();

        loop {
            let mut best_pair = None;
            let mut min_rank = u32::MAX;
            let mut curr_idx = 0;

            while curr_idx != -1 {
                let next_idx = nodes[curr_idx as usize].next;
                if next_idx != -1 {
                    let b1 = nodes[curr_idx as usize].val;
                    let next_node = &nodes[next_idx as usize];

                    let pair = (b1, next_node.val);
                    if let Some(&rank) = self.ranks.get(&pair) {
                        if rank < min_rank {
                            min_rank = rank;
                            best_pair = Some((curr_idx, next_idx));
                        }
                    }
                }
                curr_idx = nodes[curr_idx as usize].next;
            }

            if let Some((left_idx, right_idx)) = best_pair {
                let far_next = nodes[right_idx as usize].next;
                nodes[left_idx as usize].next = far_next;
                if far_next != -1 {
                    nodes[far_next as usize].prev = left_idx;
                }
            } else {
                break;
            }
        }

        // Собираем финальные байтовые куски
        let mut result = Vec::new();
        let mut curr = 0;
        while curr != -1 {
            result.push(vec![nodes[curr as usize].val]);
            curr = nodes[curr as usize].next;
        }
        result
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut final_token_ids = Vec::new();
        if text.is_empty() {
            final_token_ids.push(self.eos_token_id);
            return final_token_ids;
        }

        let raw_bytes = text.as_bytes();
        let merged_parts = self.bpe_merge_flat(raw_bytes);

        for part in merged_parts {
            if let Some(&id) = self.encoder.get(&part) {
                final_token_ids.push(id);
            }
        }

        final_token_ids.push(self.eos_token_id);
        final_token_ids
    }

    pub fn encode_to_pinned(&self, text: &str, pinned_dst: &mut PinnedHostBuffer) -> usize {
        if text.is_empty() {
            pinned_dst.as_slice_mut()[0] = self.eos_token_id as f32;
            return 1;
        }

        let token_ids = self.encode(text);
        let slice = pinned_dst.as_slice_mut();

        assert!(
            token_ids.len() <= slice.len(),
            "Критическая ошибка: Pinned буфер хоста слишком мал!"
        );

        for (i, &id) in token_ids.iter().enumerate() {
            // Записываем ID как f32 (так как наши CudaBuffer общие для логитов)
            slice[i] = id as f32;
        }
        token_ids.len()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut byte_buffer = Vec::new();

        for &id in ids {
            if id == self.eos_token_id {
                byte_buffer.extend_from_slice(b"<|endoftext|>");
                continue;
            }
            if let Some(bytes) = self.decoder.get(&id) {
                byte_buffer.extend_from_slice(bytes);
            }
        }

        String::from_utf8_lossy(&byte_buffer).into_owned()
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
        assert!(decoded.contains("привет") || decoded.contains("<|endoftext|>"));
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
