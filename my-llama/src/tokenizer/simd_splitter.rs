pub struct SimdSplitter;

#[repr(align(32))]
struct AlignedLookup {
    data: [i8; 32],
}

static LOOKUP_MASK: AlignedLookup = AlignedLookup {
    data: [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    ],
};

impl SimdSplitter {
    #[inline(always)]
    pub fn split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::split_avx2(text, ids_buffer, byte_fallback) };
            }
        }
        Self::split_fallback(text, ids_buffer, byte_fallback)
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn split_avx2(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
        use std::arch::x86_64::*;

        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut tokens_found = 0;
        let mut idx = 0;

        let buffer_len = ids_buffer.len();
        let ids_ptr = ids_buffer.as_mut_ptr();
        let fallback_ptr = byte_fallback.as_ptr();

        let lookup_mask = unsafe { _mm256_load_si256(LOOKUP_MASK.data.as_ptr() as *const __m256i) };
        let low_nibble_mask = _mm256_set1_epi8(0x0F);
        let non_printable_mask = _mm256_set1_epi8(33);

        let mut arr0 = [0u8; 32];
        let mut arr1 = [0u8; 32];
        let mut arr2 = [0u8; 32];
        let mut arr3 = [0u8; 32];

        while idx + 128 <= len {
            let base_ptr = unsafe { bytes.as_ptr().add(idx) };

            _mm_prefetch(unsafe { base_ptr.add(256) as *const i8 }, _MM_HINT_T0);

            let chunk0 = unsafe { _mm256_loadu_si256(base_ptr as *const __m256i) };
            let chunk1 = unsafe { _mm256_loadu_si256(base_ptr.add(32) as *const __m256i) };
            let chunk2 = unsafe { _mm256_loadu_si256(base_ptr.add(64) as *const __m256i) };
            let chunk3 = unsafe { _mm256_loadu_si256(base_ptr.add(96) as *const __m256i) };

            unsafe {
                _mm256_storeu_si256(arr0.as_mut_ptr() as *mut __m256i, chunk0);
                _mm256_storeu_si256(arr1.as_mut_ptr() as *mut __m256i, chunk1);
                _mm256_storeu_si256(arr2.as_mut_ptr() as *mut __m256i, chunk2);
                _mm256_storeu_si256(arr3.as_mut_ptr() as *mut __m256i, chunk3);
            }

            let low0 = _mm256_and_si256(chunk0, low_nibble_mask);
            let low1 = _mm256_and_si256(chunk1, low_nibble_mask);
            let low2 = _mm256_and_si256(chunk2, low_nibble_mask);
            let low3 = _mm256_and_si256(chunk3, low_nibble_mask);

            let shuf0 = _mm256_shuffle_epi8(lookup_mask, low0);
            let shuf1 = _mm256_shuffle_epi8(lookup_mask, low1);
            let shuf2 = _mm256_shuffle_epi8(lookup_mask, low2);
            let shuf3 = _mm256_shuffle_epi8(lookup_mask, low3);

            let low_test0 = _mm256_cmpgt_epi8(non_printable_mask, chunk0);
            let low_test1 = _mm256_cmpgt_epi8(non_printable_mask, chunk1);
            let low_test2 = _mm256_cmpgt_epi8(non_printable_mask, chunk2);
            let low_test3 = _mm256_cmpgt_epi8(non_printable_mask, chunk3);

            let delim0 = _mm256_and_si256(shuf0, low_test0);
            let delim1 = _mm256_and_si256(shuf1, low_test1);
            let delim2 = _mm256_and_si256(shuf2, low_test2);
            let delim3 = _mm256_and_si256(shuf3, low_test3);

            let mask0 = !_mm256_movemask_epi8(delim0) as u32;
            let mask1 = !_mm256_movemask_epi8(delim1) as u32;
            let mask2 = !_mm256_movemask_epi8(delim2) as u32;
            let mask3 = !_mm256_movemask_epi8(delim3) as u32;

            macro_rules! process_mask_array {
                ($mask:ident, $arr:ident) => {{
                    let mut m = $mask;
                    while m != 0 {
                        let tz = m.trailing_zeros() as usize;

                        if tokens_found >= buffer_len {
                            return tokens_found;
                        }

                        unsafe {
                            let byte_val = *$arr.get_unchecked(tz);
                            *ids_ptr.add(tokens_found) = *fallback_ptr.add(byte_val as usize);
                        }
                        tokens_found += 1;
                        m &= m - 1;
                    }
                }};
            }

            process_mask_array!(mask0, arr0);
            process_mask_array!(mask1, arr1);
            process_mask_array!(mask2, arr2);
            process_mask_array!(mask3, arr3);

            idx += 128;
        }

        while idx < len {
            let b = unsafe { *bytes.as_ptr().add(idx) };
            let is_delim = if b <= 32 { ((1u64 << b) & 0x10000000600u64) != 0 } else { false };

            if !is_delim {
                if tokens_found >= buffer_len {
                    return tokens_found;
                }
                unsafe {
                    *ids_ptr.add(tokens_found) = *fallback_ptr.add(b as usize);
                }
                tokens_found += 1;
            }
            idx += 1;
        }

        tokens_found
    }

    #[inline(always)]
    fn split_fallback(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut tokens_found = 0;

        for idx in 0..len {
            let b = unsafe { *bytes.get_unchecked(idx) };
            let is_delim = if b <= 32 { ((1u64 << b) & 0x10000000600u64) != 0 } else { false };
            if !is_delim {
                if tokens_found >= ids_buffer.len() {
                    break;
                }
                unsafe {
                    *ids_buffer.get_unchecked_mut(tokens_found) = *byte_fallback.get_unchecked(b as usize);
                }
                tokens_found += 1;
            }
        }
        tokens_found
    }
}
