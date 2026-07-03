use std::arch::x86_64::*;

const ASCII_LF: i8 = 10;
const ASCII_CR: i8 = 13;
const ASCII_TAB: i8 = 9;
const ASCII_SPACE: i8 = 32;

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

    let v_lf = _mm256_set1_epi8(ASCII_LF);
    let v_cr = _mm256_set1_epi8(ASCII_CR);
    let v_tab = _mm256_set1_epi8(ASCII_TAB);
    let v_space = _mm256_set1_epi8(ASCII_SPACE);

    let v_digit_min = _mm256_set1_epi8(0x30);
    let v_digit_max = _mm256_set1_epi8(0x39);
    let v_sign_bit = _mm256_set1_epi8(i8::MIN);

    let mut current_token_start = 0u32;

    #[inline(always)]
    fn get_byte_category(b: u8) -> u8 {
        if b == 10 || b == 13 || b == 9 {
            0 // Жесткий разделитель (перенос строки/таб)
        } else if b == 32 {
            1 // Пробел (может приклеиваться к слову)
        } else if b >= 48 && b <= 57 {
            2 // Цифра
        } else if b >= 128 {
            3 // Кириллица / Мультибайт (буквы)
        } else if (b >= 65 && b <= 90) || (b >= 97 && b <= 122) {
            3 // ASCII Буквы
        } else {
            4 // Знаки препинания, скобки и прочие спецсимволы
        }
    }

    let mut current_category = get_byte_category(unsafe { *bytes.get_unchecked(0) });

    while idx + 32 <= len {
        let base_ptr = unsafe { bytes.as_ptr().add(idx) };
        let chunk = unsafe { _mm256_loadu_si256(base_ptr as *const __m256i) };

        unsafe {
            let cmp_lf = _mm256_cmpeq_epi8(chunk, v_lf);
            let cmp_cr = _mm256_cmpeq_epi8(chunk, v_cr);
            let cmp_tab = _mm256_cmpeq_epi8(chunk, v_tab);
            let mask_sep = _mm256_or_si256(_mm256_or_si256(cmp_lf, cmp_cr), cmp_tab);

            let mask_space = _mm256_cmpeq_epi8(chunk, v_space);

            let chunk_unsigned = _mm256_xor_si256(chunk, v_sign_bit);
            let digit_min_u = _mm256_xor_si256(v_digit_min, v_sign_bit);
            let digit_max_u = _mm256_xor_si256(v_digit_max, v_sign_bit);
            let mask_digit = _mm256_and_si256(
                _mm256_cmpgt_epi8(chunk_unsigned, digit_min_u),
                _mm256_cmpgt_epi8(digit_max_u, chunk_unsigned),
            );
            // Добираем точные границы '0' и '9'
            let mask_digit = _mm256_or_si256(mask_digit, _mm256_cmpeq_epi8(chunk, v_digit_min));
            let mask_digit = _mm256_or_si256(mask_digit, _mm256_cmpeq_epi8(chunk, v_digit_max));

            // 4. Ищем буквы: кириллица (знаковый бит установлен) + ASCII буквы
            let mask_cyrillic = _mm256_cmpgt_epi8(v_space, chunk); // b >= 128 интерпретируется как отрицательное число

            // ASCII Буквы: 'A'..='Z' (65..90) и 'a'..='z' (97..122)
            let v_a_min = _mm256_set1_epi8(64);
            let v_z_max = _mm256_set1_epi8(91);
            let v_a_low = _mm256_set1_epi8(96);
            let v_z_low = _mm256_set1_epi8(123);
            let mask_ascii_caps = _mm256_and_si256(_mm256_cmpgt_epi8(chunk, v_a_min), _mm256_cmpgt_epi8(v_z_max, chunk));
            let mask_ascii_low = _mm256_and_si256(_mm256_cmpgt_epi8(chunk, v_a_low), _mm256_cmpgt_epi8(v_z_low, chunk));
            let mask_letters = _mm256_or_si256(_mm256_or_si256(mask_cyrillic, mask_ascii_caps), mask_ascii_low);

            // Сканируем маски побитово для всех 32 элементов в регистре YMM
            let m_sep = _mm256_movemask_epi8(mask_sep) as u32;
            let m_space = _mm256_movemask_epi8(mask_space) as u32;
            let m_digit = _mm256_movemask_epi8(mask_digit) as u32;
            let m_letters = _mm256_movemask_epi8(mask_letters) as u32;

            for bit_idx in 0..32 {
                let bit = 1 << bit_idx;

                // Вычисляем категорию текущего байта на основе SIMD масок
                let cat = if (m_sep & bit) != 0 {
                    0
                } else if (m_space & bit) != 0 {
                    1
                } else if (m_digit & bit) != 0 {
                    2
                } else if (m_letters & bit) != 0 {
                    3
                } else {
                    4
                };

                // ОФИЦИАЛЬНОЕ ПРАВИЛО BPE СЛИЯНИЙ ПРОБЕЛОВ:
                // Если предыдущий символ был пробелом (cat == 1), а текущий — буква (cat == 3),
                // мы НЕ ДЕЛАЕМ разрез чанка, а приклеиваем этот пробел к началу слова!
                let is_bpe_space_glue = current_category == 1 && cat == 3;

                if cat != current_category && !is_bpe_space_glue {
                    let current_global_idx = (idx + bit_idx) as u32;

                    if token_count < max_tokens {
                        *offsets_buffer.get_unchecked_mut(token_count) = current_token_start;
                        *len_buffer.get_unchecked_mut(token_count) = current_global_idx - current_token_start;
                        token_count += 1;
                    }

                    current_token_start = current_global_idx;
                }
                current_category = cat;
            }
        }
        idx += 32;
    }

    while idx < len {
        let b = unsafe { *bytes.get_unchecked(idx) };
        let cat = get_byte_category(b);

        let is_bpe_space_glue = current_category == 1 && cat == 3;

        if cat != current_category && !is_bpe_space_glue {
            let current_global_idx = idx as u32;
            if token_count < max_tokens {
                unsafe {
                    *offsets_buffer.get_unchecked_mut(token_count) = current_token_start;
                    *len_buffer.get_unchecked_mut(token_count) = current_global_idx - current_token_start;
                }
                token_count += 1;
            }
            current_token_start = current_global_idx;
        }
        current_category = cat;
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
