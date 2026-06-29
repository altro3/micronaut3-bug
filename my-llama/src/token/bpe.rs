use base64::{prelude::BASE64_STANDARD, Engine};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

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

    fn bpe_merge(&self, word_bytes: Vec<u8>) -> Vec<Vec<u8>> {
        let mut parts: Vec<Vec<u8>> = word_bytes.iter().map(|&b| vec![b]).collect();

        loop {
            if parts.len() < 2 {
                break;
            }

            let mut best_pair = None;
            let mut min_rank = u32::MAX;

            for i in 0..parts.len() - 1 {
                if parts[i].len() == 1 && parts[i + 1].len() == 1 {
                    let b1 = parts[i][0];
                    let b2 = parts[i + 1][0];
                    let pair = (b1, b2);

                    if let Some(&rank) = self.ranks.get(&pair) {
                        if rank < min_rank {
                            min_rank = rank;
                            best_pair = Some((i, pair));
                        }
                    }
                }
            }

            if let Some((idx, _pair)) = best_pair {
                let mut first = parts.remove(idx);
                let second = parts.remove(idx);
                first.extend(second);
                parts.insert(idx, first);
            } else {
                break;
            }
        }
        parts
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut final_token_ids = Vec::new();
        if text.is_empty() {
            final_token_ids.push(self.eos_token_id);
            return final_token_ids;
        }

        let raw_bytes = text.as_bytes().to_vec();
        let merged_parts = self.bpe_merge(raw_bytes);

        for part in merged_parts {
            if let Some(&id) = self.encoder.get(&part) {
                final_token_ids.push(id);
            }
        }

        final_token_ids.push(self.eos_token_id);
        final_token_ids
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
