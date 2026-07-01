use crate::cuda::PinnedHostBuffer;
use crate::token::BpePair;
use serde_json::{from_reader, Value};
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};

pub struct BpeTokenizer {
    encoder_pool: HashMap<u64, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    pub eos_token_id: u32,
    byte_fallback: [u32; 256],
}

impl BpeTokenizer {
    pub fn from_file(file_path: &str) -> std::io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let json_data: Value = from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        let mut encoder_pool = HashMap::new();
        let mut decoder = HashMap::new();
        let mut byte_fallback = [0u32; 256];

        let vocab = json_data["model"]["vocab"]
            .as_object()
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "Не найден блок 'model.vocab'"))?;

        for (token_str, id_val) in vocab {
            let id = id_val.as_u64().ok_or_else(|| Error::new(ErrorKind::InvalidData, "Format ID error"))? as u32;
            let token_bytes = token_str.as_bytes().to_vec();

            if token_bytes.len() == 1 {
                byte_fallback[token_bytes[0] as usize] = id;
            }

            let hash = Self::hash_bytes(&token_bytes);
            encoder_pool.insert(hash, id);
            decoder.insert(id, token_bytes);
        }

        for b in 0..=255 {
            if byte_fallback[b] == 0 {
                let hash = Self::hash_bytes(&[b as u8]);
                byte_fallback[b] = *encoder_pool.get(&hash).unwrap_or(&(b as u32));
            }
        }

        let mut eos_id = 151643;
        if let Some(added_tokens) = json_data["added_tokens"].as_array() {
            for token in added_tokens {
                if token["content"].as_str() == Some("<|endoftext|") {
                    if let Some(id) = token["id"].as_u64() {
                        eos_id = id as u32;
                    }
                }
            }
        }

        println!("[API] Кэш BPE собран. Токенов: {}, EOS: {}", encoder_pool.len(), eos_id);
        Ok(BpeTokenizer {
            encoder_pool,
            decoder,
            eos_token_id: eos_id,
            byte_fallback,
        })
    }

    #[inline(always)]
    fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    #[inline(always)]
    fn get_pair_rank(&self, left: usize, right: usize, len: usize, next: &[usize], raw_bytes: &[u8]) -> Option<u32> {
        if right >= len {
            return None;
        }
        let end = next[right];
        let hash = Self::hash_bytes(&raw_bytes[left..end]);
        self.encoder_pool.get(&hash).copied()
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
        let len = raw_bytes.len();

        let mut prev = vec![0usize; len + 1];
        let mut next = vec![0usize; len + 1];
        let mut heap = BinaryHeap::with_capacity(len);

        for i in 0..len {
            prev[i] = i.wrapping_sub(1);
            next[i] = i + 1;
        }
        next[len] = len;

        for i in 0..(len - 1) {
            if let Some(rank) = self.get_pair_rank(i, i + 1, len, &next, raw_bytes) {
                heap.push(BpePair { rank, index: i });
            }
        }

        while let Some(BpePair { rank, index: l }) = heap.pop() {
            let r = next[l];
            if r >= len || next[r] == l {
                continue;
            }

            if let Some(current_rank) = self.get_pair_rank(l, r, len, &next, raw_bytes) {
                if current_rank != rank {
                    continue;
                }

                let after_r = next[r];
                next[l] = after_r;
                if after_r < len {
                    prev[after_r] = l;
                }

                let before_l = prev[l];
                if before_l != usize::MAX {
                    if let Some(new_rank) = self.get_pair_rank(before_l, l, len, &next, raw_bytes) {
                        heap.push(BpePair {
                            rank: new_rank,
                            index: before_l,
                        });
                    }
                }
                if after_r < len {
                    if let Some(new_rank) = self.get_pair_rank(l, after_r, len, &next, raw_bytes) {
                        heap.push(BpePair { rank: new_rank, index: l });
                    }
                }
            }
        }

        let mut token_count = 0;
        let mut i = 0;
        while i < len {
            if token_count >= slice.len() {
                break;
            }
            let end = next[i];
            let token_bytes = &raw_bytes[i..end];

            let token_id = if let Some(&id) = self.encoder_pool.get(&Self::hash_bytes(token_bytes)) {
                id
            } else if token_bytes.len() == 1 {
                self.byte_fallback[token_bytes[0] as usize]
            } else {
                token_bytes[0] as u32
            };

            slice[token_count] = token_id as f32;
            token_count += 1;
            i = end;
        }
        token_count
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut dummy = PinnedHostBuffer::new(text.len() + 1);
        let count = self.encode_to_pinned(text, &mut dummy);
        dummy.as_slice_mut()[..count].iter().map(|&x| x as u32).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut byte_buffer = Vec::with_capacity(ids.len() * 4);
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
}

unsafe impl Send for BpeTokenizer {}
unsafe impl Sync for BpeTokenizer {}
