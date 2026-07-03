use std::arch::x86_64::*;

#[repr(align(32))]
struct AlignedLookup {
    data: [i8; 32],
}

static LOOKUP_MASK: AlignedLookup = AlignedLookup {
    data: [
        -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0,
    ],
};

#[target_feature(enable = "avx2,bmi2")]
pub unsafe fn split(text: &str, offsets_buffer: &mut [u32], len_buffer: &mut [u32]) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut idx = 0;
    let mut token_count = 0;
    let max_tokens = offsets_buffer.len();

    let lookup_mask = unsafe { _mm256_load_si256((&LOOKUP_MASK.data as *const [i8; 32]) as *const __m256i) };
    let low_nibble_mask = _mm256_set1_epi8(0x0F);

    let non_printable_mask_unsigned = _mm256_set1_epi8((33u8 ^ 0x80u8) as i8);
    let sign_bit = _mm256_set1_epi8(i8::MIN);

    let mut was_in_word = false;
    let mut current_word_start = 0u32;

    while idx + 32 <= len {
        let base_ptr = unsafe { bytes.as_ptr().add(idx) };
        let chunk = unsafe { _mm256_loadu_si256(base_ptr as *const __m256i) };

        let chunk_unsigned = _mm256_xor_si256(chunk, sign_bit);
        let is_less_than_33 = _mm256_cmpgt_epi8(non_printable_mask_unsigned, chunk_unsigned);

        let delim = _mm256_and_si256(
            _mm256_shuffle_epi8(lookup_mask, _mm256_and_si256(chunk, low_nibble_mask)),
            is_less_than_33,
        );

        let delim_mask = _mm256_movemask_epi8(delim) as u32;
        let valid_mask = delim_mask ^ 0xFFFFFFFF;

        let mut bits = valid_mask;
        let mut bit_offset = 0;

        while bits != 0 {
            let tz = bits.trailing_zeros() as usize;

            bits >>= tz;
            bit_offset += tz;

            let run_len = (!bits).trailing_zeros() as usize;

            if !was_in_word {
                current_word_start = (idx + bit_offset) as u32;
            }

            let word_end = idx + bit_offset + run_len;

            if word_end < idx + 32 || (word_end == idx + 32 && (delim_mask & (1 << 31)) != 0) {
                if token_count < max_tokens {
                    unsafe {
                        *offsets_buffer.get_unchecked_mut(token_count) = current_word_start;
                        *len_buffer.get_unchecked_mut(token_count) = (word_end as u32) - current_word_start;
                    }
                    token_count += 1;
                }
                was_in_word = false;
            } else {
                was_in_word = true;
            }

            bits >>= run_len;
            bit_offset += run_len;
        }

        if (valid_mask & (1 << 31)) == 0 && was_in_word {
            if token_count < max_tokens {
                unsafe {
                    *offsets_buffer.get_unchecked_mut(token_count) = current_word_start;
                    *len_buffer.get_unchecked_mut(token_count) = ((idx + 32) as u32) - current_word_start;
                }
                token_count += 1;
            }
            was_in_word = false;
        }

        idx += 32;
    }

    while idx < len {
        let b = unsafe { *bytes.get_unchecked(idx) };
        let is_delim = b == 32 || b == 10 || b == 9 || b == 13;

        if !is_delim {
            if !was_in_word {
                current_word_start = idx as u32;
                was_in_word = true;
            }
        } else if was_in_word {
            if token_count < max_tokens {
                unsafe {
                    *offsets_buffer.get_unchecked_mut(token_count) = current_word_start;
                    *len_buffer.get_unchecked_mut(token_count) = (idx as u32) - current_word_start;
                }
                token_count += 1;
            }
            was_in_word = false;
        }
        idx += 1;
    }

    if was_in_word && token_count < max_tokens {
        unsafe {
            *offsets_buffer.get_unchecked_mut(token_count) = current_word_start;
            *len_buffer.get_unchecked_mut(token_count) = (len as u32) - current_word_start;
        }
        token_count += 1;
    }

    token_count
}
