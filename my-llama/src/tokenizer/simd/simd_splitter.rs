use crate::tokenizer::simd::avx2;

pub struct SimdSplitter;

impl SimdSplitter {
    #[inline(always)]
    pub fn split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::split(text, ids_buffer, byte_fallback) };
            }
        }
        Self::fallback_split(text, ids_buffer, byte_fallback)
    }

    fn fallback_split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut tokens_found = 0;
        let bytes_ptr = bytes.as_ptr();
        let buf_len = ids_buffer.len();

        for idx in 0..len {
            let b = unsafe { *bytes_ptr.add(idx) };
            let is_delim = if b <= 32 { ((1u64 << b) & 0x10000000600u64) != 0 } else { false };
            if !is_delim {
                if tokens_found >= buf_len {
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
