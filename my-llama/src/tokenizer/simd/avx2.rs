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
pub fn split(text: &str, offsets_buffer: &mut [u32], len_buffer: &mut [u32]) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return 0;
    }

    let mut idx = 0;
    let mut token_count = 0;
    let max_tokens = offsets_buffer.len();

    let lookup_mask = unsafe { _mm256_load_si256((&LOOKUP_MASK.data as *const [i8; 32]) as *const __m256i) };
    let low_nibble_mask = _mm256_set1_epi8(0x0F);
    let non_printable_mask_unsigned = _mm256_set1_epi8((33u8 ^ 0x80u8) as i8);
    let sign_bit = _mm256_set1_epi8(i8::MIN);

    // Текущий открытый токен стартует с самого начала текста
    let mut current_token_start = 0u32;

    // Какое состояние у нас сейчас открыто: true — буквы, false — пробелы
    let mut is_current_token_letter = {
        let first = unsafe { *bytes.get_unchecked(0) };
        !(first == 32 || first == 10 || first == 9 || first == 13)
    };

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
        let valid_mask = delim_mask ^ 0xFFFFFFFF; // 1 — буква, 0 — пробел

        for bit_idx in 0..32 {
            let is_bit_letter = (valid_mask & (1 << bit_idx)) != 0;

            // Если тип символа изменился — закрываем старый токен и открываем новый!
            if is_bit_letter != is_current_token_letter {
                let current_global_idx = (idx + bit_idx) as u32;

                if token_count < max_tokens {
                    unsafe {
                        *offsets_buffer.get_unchecked_mut(token_count) = current_token_start;
                        *len_buffer.get_unchecked_mut(token_count) = current_global_idx - current_token_start;
                    }
                    token_count += 1;
                }

                current_token_start = current_global_idx;
                is_current_token_letter = is_bit_letter;
            }
        }

        idx += 32;
    }

    while idx < len {
        let b = unsafe { *bytes.get_unchecked(idx) };
        let is_letter = !(b == 32 || b == 10 || b == 9 || b == 13);

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

    // Закрываем самый последний токен всего текста
    if token_count < max_tokens {
        unsafe {
            *offsets_buffer.get_unchecked_mut(token_count) = current_token_start;
            *len_buffer.get_unchecked_mut(token_count) = (len as u32) - current_token_start;
        }
        token_count += 1;
    }

    token_count
}
