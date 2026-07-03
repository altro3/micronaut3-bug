use crate::tokenizer::simd::avx2;

pub struct SimdSplitter;

impl SimdSplitter {
    #[inline(always)]
    pub fn split(text: &str, offsets_buffer: &mut [u32], len_buffer: &mut [u32]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("bmi2") {
                return unsafe { avx2::split(text, offsets_buffer, len_buffer) };
            }
        }

        // Если AVX2 нет (или мы на ARM), запускаем безопасный базовый вариант
        Self::fallback_split(text, offsets_buffer, len_buffer)
    }

    pub fn fallback_split(text: &str, offsets_buffer: &mut [u32], len_buffer: &mut [u32]) -> usize {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let max_tokens = offsets_buffer.len();

        let mut token_count = 0;
        let mut was_in_word = false;
        let mut current_word_start = 0u32;

        for idx in 0..len {
            let b = unsafe { *bytes.as_ptr().add(idx) };
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
}
