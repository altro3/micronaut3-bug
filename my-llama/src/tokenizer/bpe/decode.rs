use crate::tokenizer::bpe_tokenizer::BpeTokenizer;

impl BpeTokenizer {
    pub fn decode(&self, tokens: &[u32]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let mut raw_buffer: Vec<u8> = Vec::with_capacity(tokens.len() * 4);
        let mut w = 0;

        let offsets_ptr = self.vocab_offsets_flat.as_ptr();
        let bytes_ptr = self.vocab_bytes_flat.as_ptr();

        // 1. Быстрая сборка плоского массива Latin-1 байт через сырые указатели словаря
        for &token_id in tokens {
            let id = token_id as usize;

            if id < self.vocab_offsets_flat.len() {
                unsafe {
                    let packed = *offsets_ptr.add(id);
                    if packed != 0 {
                        let offset = (packed >> 32) as usize;
                        let length = (packed & 0xFFFFFFFF) as usize;

                        if w + length > raw_buffer.capacity() {
                            raw_buffer.reserve(tokens.len() * 2 + length);
                        }

                        std::ptr::copy_nonoverlapping(bytes_ptr.add(offset), raw_buffer.as_mut_ptr().add(w), length);
                        w += length;
                    }
                }
            }
        }

        unsafe {
            raw_buffer.set_len(w);
            let mut i = 0;
            let mut write_idx = 0;
            let res_ptr = raw_buffer.as_mut_ptr();

            while i < w {
                let b0 = *res_ptr.add(i);
                let cp: u32;
                let step: usize;

                if b0 < 0x80 {
                    cp = b0 as u32;
                    step = 1;
                } else if (b0 & 0xE0) == 0xC0 && i + 1 < w {
                    cp = (((b0 & 0x1F) as u32) << 6) | (*res_ptr.add(i + 1) & 0x3F) as u32;
                    step = 2;
                } else if (b0 & 0xF0) == 0xE0 && i + 2 < w {
                    cp = (((b0 & 0x0F) as u32) << 12) | (((*res_ptr.add(i + 1) & 0x3F) as u32) << 6) | (*res_ptr.add(i + 2) & 0x3F) as u32;
                    step = 3;
                } else if (b0 & 0xF8) == 0xF0 && i + 3 < w {
                    cp = (((b0 & 0x07) as u32) << 18)
                        | (((*res_ptr.add(i + 1) & 0x3F) as u32) << 12)
                        | (((*res_ptr.add(i + 2) & 0x3F) as u32) << 6)
                        | (*res_ptr.add(i + 3) & 0x3F) as u32;
                    step = 4;
                } else {
                    *res_ptr.add(write_idx) = b0;
                    write_idx += 1;
                    i += 1;
                    continue;
                }

                let raw_byte = match cp {
                    0x00..=0x7F => cp as u8,
                    0x00A0..=0x00FF => cp as u8,
                    0x0100..=0x011F => (cp - 0x0100) as u8,
                    0x0120..=0x013F => (cp - 0x0120 + 127) as u8,
                    0x0140 => 173,
                    _ => b0,
                };

                if cp >= 0x0100 && cp <= 0x01FF {
                    *res_ptr.add(write_idx) = raw_byte;
                    write_idx += 1;
                    i += step;
                } else {
                    for s in 0..step {
                        *res_ptr.add(write_idx + s) = *res_ptr.add(i + s);
                    }
                    write_idx += step;
                    i += step;
                }
            }

            raw_buffer.set_len(write_idx);
            String::from_utf8_unchecked(raw_buffer)
        }
    }
}
