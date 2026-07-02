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
        let text_bytes = text.as_bytes();
        let total_length = text_bytes.len();
        let mut current_token_start = 0;
        let mut tokens_found = 0;

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                current_token_start = unsafe { Self::split_avx2(text_bytes, total_length, tokens_buffer, &mut tokens_found) };
            }
        }

        let mut current_index = current_token_start;
        while current_index < total_length {
            let current_byte = unsafe { *text_bytes.get_unchecked(current_index) };

            if current_byte == b' ' || current_byte == b'\n' || current_byte == b'\r' || current_byte == b'\t' {
                if current_index > current_token_start {
                    if tokens_found < tokens_buffer.len() {
                        unsafe {
                            *tokens_buffer.get_unchecked_mut(tokens_found) = TokenSpan {
                                start: current_token_start as u32,
                                end: current_index as u32,
                            };
                        }
                        tokens_found += 1;
                    } else {
                        return tokens_found;
                    }
                }
                current_token_start = current_index + 1;
            }
            current_index += 1;
        }

        if current_token_start < total_length && tokens_found < tokens_buffer.len() {
            unsafe {
                *tokens_buffer.get_unchecked_mut(tokens_found) = TokenSpan {
                    start: current_token_start as u32,
                    end: total_length as u32,
                };
            }
            tokens_found += 1;
        }

        tokens_found
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn split_avx2(text_bytes: &[u8], total_length: usize, tokens_buffer: &mut [TokenSpan], tokens_found: &mut usize) -> usize {
        use std::arch::x86_64::*;

        let mut current_index = 0;
        let mut current_token_start = 0;

        let vector_space = _mm256_set1_epi8(b' ' as i8);
        let vector_newline = _mm256_set1_epi8(b'\n' as i8);
        let vector_carriage = _mm256_set1_epi8(b'\r' as i8);
        let vector_tab = _mm256_set1_epi8(b'\t' as i8);

        macro_rules! consume_bitmask {
            ($bitmask:expr, $chunk_offset:expr) => {{
                let mask = $bitmask;
                let mut success = true;

                if mask != 0 {
                    if (mask & (mask - 1)) == 0 {
                        let trailing_zeros = mask.trailing_zeros() as usize;
                        let split_position = $chunk_offset + trailing_zeros;

                        if split_position > current_token_start {
                            if *tokens_found < tokens_buffer.len() {
                                unsafe {
                                    *tokens_buffer.get_unchecked_mut(*tokens_found) = TokenSpan {
                                        start: current_token_start as u32,
                                        end: split_position as u32,
                                    };
                                }
                                *tokens_found += 1;
                            } else {
                                success = false;
                            }
                        }
                        current_token_start = split_position + 1;
                    } else {
                        let mut temp_mask = mask;
                        while temp_mask != 0 {
                            let trailing_zeros = temp_mask.trailing_zeros() as usize;
                            let split_position = $chunk_offset + trailing_zeros;

                            if split_position > current_token_start {
                                if *tokens_found < tokens_buffer.len() {
                                    unsafe {
                                        *tokens_buffer.get_unchecked_mut(*tokens_found) = TokenSpan {
                                            start: current_token_start as u32,
                                            end: split_position as u32,
                                        };
                                    }
                                    *tokens_found += 1;
                                } else {
                                    success = false;
                                    break;
                                }
                            }
                            current_token_start = split_position + 1;
                            temp_mask &= temp_mask - 1;
                        }
                    }
                }
                success
            }};
        }

        while current_index + 64 <= total_length {
            let delimiters_bitmask_1;
            let delimiters_bitmask_2;

            unsafe {
                let base_pointer = text_bytes.as_ptr().add(current_index);

                let memory_chunk_1 = _mm256_loadu_si256(base_pointer as *const __m256i);
                let memory_chunk_2 = _mm256_loadu_si256(base_pointer.add(32) as *const __m256i);

                let match_space_1 = _mm256_cmpeq_epi8(memory_chunk_1, vector_space);
                let match_newline_1 = _mm256_cmpeq_epi8(memory_chunk_1, vector_newline);
                let match_carriage_1 = _mm256_cmpeq_epi8(memory_chunk_1, vector_carriage);
                let match_tab_1 = _mm256_cmpeq_epi8(memory_chunk_1, vector_tab);
                let combined_matches_1 = _mm256_or_si256(
                    _mm256_or_si256(match_space_1, match_newline_1),
                    _mm256_or_si256(match_carriage_1, match_tab_1),
                );
                delimiters_bitmask_1 = _mm256_movemask_epi8(combined_matches_1) as u32;

                let match_space_2 = _mm256_cmpeq_epi8(memory_chunk_2, vector_space);
                let match_newline_2 = _mm256_cmpeq_epi8(memory_chunk_2, vector_newline);
                let match_carriage_2 = _mm256_cmpeq_epi8(memory_chunk_2, vector_carriage);
                let match_tab_2 = _mm256_cmpeq_epi8(memory_chunk_2, vector_tab);
                let combined_matches_2 = _mm256_or_si256(
                    _mm256_or_si256(match_space_2, match_newline_2),
                    _mm256_or_si256(match_carriage_2, match_tab_2),
                );
                delimiters_bitmask_2 = _mm256_movemask_epi8(combined_matches_2) as u32;
            }

            if !consume_bitmask!(delimiters_bitmask_1, current_index) {
                return current_token_start;
            }
            if !consume_bitmask!(delimiters_bitmask_2, current_index + 32) {
                return current_token_start;
            }

            current_index += 64;
        }

        while current_index + 32 <= total_length {
            let delimiters_bitmask;

            unsafe {
                let memory_chunk = _mm256_loadu_si256(text_bytes.as_ptr().add(current_index) as *const __m256i);
                let match_space = _mm256_cmpeq_epi8(memory_chunk, vector_space);
                let match_newline = _mm256_cmpeq_epi8(memory_chunk, vector_newline);
                let match_carriage = _mm256_cmpeq_epi8(memory_chunk, vector_carriage);
                let match_tab = _mm256_cmpeq_epi8(memory_chunk, vector_tab);

                let combined_matches = _mm256_or_si256(_mm256_or_si256(match_space, match_newline), _mm256_or_si256(match_carriage, match_tab));
                delimiters_bitmask = _mm256_movemask_epi8(combined_matches) as u32;
            }

            if !consume_bitmask!(delimiters_bitmask, current_index) {
                return current_token_start;
            }
            current_index += 32;
        }

        current_token_start
    }
}
