use crate::tokenizer::BpeTokenizer;
use std::ptr::copy_nonoverlapping;

impl BpeTokenizer {
    pub fn decode(&self, tokens: &[u32]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let mut raw_buffer: Vec<u8> = Vec::with_capacity(tokens.len() * 8);
        let mut w = 0;

        let offsets_ptr = self.vocab_offsets_flat.as_ptr();
        let bytes_ptr = self.vocab_bytes_flat.as_ptr();

        for &token_id in tokens {
            let id = token_id as usize;

            if id < self.vocab_offsets_flat.len() {
                unsafe {
                    let packed = *offsets_ptr.add(id);
                    if packed != 0 {
                        let offset = (packed >> 32) as usize;
                        let length = (packed & 0xFFFFFFFF) as usize;

                        if w + length > raw_buffer.capacity() {
                            raw_buffer.reserve(tokens.len() * 4 + length);
                        }

                        copy_nonoverlapping(bytes_ptr.add(offset), raw_buffer.as_mut_ptr().add(w), length);
                        w += length;
                    }
                }
            }
            if id == 248076 {
                if w + 1 > raw_buffer.capacity() {
                    raw_buffer.reserve(tokens.len() * 4 + 1);
                }
                unsafe {
                    *raw_buffer.as_mut_ptr().add(w) = b' ';
                }
                w += 1;
                continue;
            }
        }

        unsafe {
            raw_buffer.set_len(w);
        }

        let raw_str = String::from_utf8_lossy(&raw_buffer);
        let mut clean_str = raw_str.replace('Ġ', " ");
        if let Some(garbage_idx) = clean_str.find('\u{FFFD}') {
            clean_str.truncate(garbage_idx);
        }

        clean_str
    }
}
