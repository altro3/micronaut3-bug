use std::arch::x86_64::*;

#[target_feature(enable = "avx2,bmi2")]
pub fn split(text: &str, offsets_buffer: &mut [u32], len_buffer: &mut [u32]) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return 0;
    }

    let mut idx = 0;
    let mut token_count = 0;
    let max_tokens = offsets_buffer.len();

    let v_newline = _mm256_set1_epi8(10);
    let v_carriage = _mm256_set1_epi8(13);

    let mut current_token_start = 0u32;

    let mut is_current_token_letter = {
        let first = unsafe { *bytes.get_unchecked(0) };
        !(first == 10 || first == 13)
    };

    while idx + 32 <= len {
        let base_ptr = unsafe { bytes.as_ptr().add(idx) };
        let chunk = unsafe { _mm256_loadu_si256(base_ptr as *const __m256i) };

        unsafe {
            let cmp_newline = _mm256_cmpeq_epi8(chunk, v_newline);
            let cmp_carriage = _mm256_cmpeq_epi8(chunk, v_carriage);

            let delim = _mm256_or_si256(cmp_newline, cmp_carriage);
            let delim_mask = _mm256_movemask_epi8(delim) as u32;
            let valid_mask = delim_mask ^ 0xFFFFFFFF;

            for bit_idx in 0..32 {
                let is_bit_letter = (valid_mask & (1 << bit_idx)) != 0;

                if is_bit_letter != is_current_token_letter {
                    let current_global_idx = (idx + bit_idx) as u32;

                    if token_count < max_tokens {
                        *offsets_buffer.get_unchecked_mut(token_count) = current_token_start;
                        *len_buffer.get_unchecked_mut(token_count) = current_global_idx - current_token_start;
                        token_count += 1;
                    }

                    current_token_start = current_global_idx;
                    is_current_token_letter = is_bit_letter;
                }
            }
        }
        idx += 32;
    }

    while idx < len {
        let b = unsafe { *bytes.get_unchecked(idx) };
        let is_letter = !(b == 10 || b == 13);

        if is_letter != is_current_token_letter {
            let current_global_idx = idx as u32;
            if token_count < max_tokens {
                unsafe {
                    *offsets_buffer.get_unchecked_mut(token_count) = current_token_start;
                    *len_buffer.get_unchecked_mut(token_count) = current_global_idx - current_token_start;
                }
                token_count += 1;
            }
            current_token_start = current_global_idx;
            is_current_token_letter = is_letter;
        }
        idx += 1;
    }

    if token_count < max_tokens {
        unsafe {
            *offsets_buffer.get_unchecked_mut(token_count) = current_token_start;
            *len_buffer.get_unchecked_mut(token_count) = (len as u32) - current_token_start;
        }
        token_count += 1;
    }

    token_count
}
