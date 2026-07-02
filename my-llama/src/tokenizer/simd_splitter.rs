pub struct SimdSplitter;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TokenSpan {
    pub start: u32,
    pub end: u32,
}

impl SimdSplitter {
    #[inline(always)]
    pub fn split(text: &str, tokens_buffer: &mut [TokenSpan]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::split_avx2(text, tokens_buffer) };
            }
        }

        Self::split_fallback(text, tokens_buffer)
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn split_avx2(text: &str, tokens_buffer: &mut [TokenSpan]) -> usize {
        use std::arch::x86_64::*;

        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut token_start: u32 = 0;
        let mut tokens_found = 0;
        let mut idx = 0;

        let buffer_len = tokens_buffer.len();
        let buf_ptr = tokens_buffer.as_mut_ptr();

        let lookup_mask = _mm256_setr_epi8(
            1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, // 0..15
            1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, // 16..31
        );
        let low_nibble_mask = _mm256_set1_epi8(0x0F);
        let non_printable_mask = _mm256_set1_epi8(33);

        while idx + 32 <= len {
            let mut bitmask = unsafe {
                let base_ptr = bytes.as_ptr().add(idx);
                let chunk = _mm256_loadu_si256(base_ptr as *const __m256i);

                let low_nibbles = _mm256_and_si256(chunk, low_nibble_mask);
                let shuffled = _mm256_shuffle_epi8(lookup_mask, low_nibbles);

                let is_low = _mm256_cmpgt_epi8(non_printable_mask, chunk);
                let is_delimiter = _mm256_and_si256(shuffled, is_low);

                _mm256_movemask_epi8(is_delimiter) as u32
            };

            while bitmask != 0 {
                let tz = bitmask.trailing_zeros() as usize;
                let split_pos = (idx + tz) as u32;

                let is_valid_token = split_pos > token_start;
                let has_space = tokens_found < buffer_len;

                if is_valid_token & has_space {
                    unsafe {
                        *buf_ptr.add(tokens_found) = TokenSpan {
                            start: token_start,
                            end: split_pos,
                        };
                    }
                    tokens_found += 1;
                }

                token_start = split_pos + 1;
                bitmask &= bitmask - 1;
            }

            idx += 32;
        }

        while idx < len {
            let b = unsafe { *bytes.get_unchecked(idx) };
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                let split_pos = idx as u32;
                if split_pos > token_start && tokens_found < buffer_len {
                    unsafe {
                        *buf_ptr.add(tokens_found) = TokenSpan {
                            start: token_start,
                            end: split_pos,
                        };
                    }
                    tokens_found += 1;
                }
                token_start = split_pos + 1;
            }
            idx += 1;
        }

        if token_start < len as u32 && tokens_found < buffer_len {
            unsafe {
                *buf_ptr.add(tokens_found) = TokenSpan {
                    start: token_start,
                    end: len as u32,
                };
            }
            tokens_found += 1;
        }

        tokens_found
    }

    #[inline(always)]
    fn split_fallback(text: &str, tokens_buffer: &mut [TokenSpan]) -> usize {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut token_start = 0;
        let mut tokens_found = 0;

        for idx in 0..len {
            let b = unsafe { *bytes.get_unchecked(idx) };
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                if idx > token_start {
                    if tokens_found >= tokens_buffer.len() {
                        return tokens_found;
                    }
                    unsafe {
                        *tokens_buffer.get_unchecked_mut(tokens_found) = TokenSpan {
                            start: token_start as u32,
                            end: idx as u32,
                        };
                    }
                    tokens_found += 1;
                }
                token_start = idx + 1;
            }
        }

        if token_start < len && tokens_found < tokens_buffer.len() {
            unsafe {
                *tokens_buffer.get_unchecked_mut(tokens_found) = TokenSpan {
                    start: token_start as u32,
                    end: len as u32,
                };
            }
            tokens_found += 1;
        }

        tokens_found
    }
}
