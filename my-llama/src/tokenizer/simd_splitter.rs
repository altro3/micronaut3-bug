use crate::tokenizer::arch::{avx2, avx512, fallback};

pub struct SimdSplitter;

impl SimdSplitter {
    #[inline(always)]
    pub fn split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                return unsafe { avx512::split(text, ids_buffer, byte_fallback) };
            }
            if is_x86_feature_detected!("avx2") {
                return unsafe { avx2::split(text, ids_buffer, byte_fallback) };
            }
        }
        fallback::split(text, ids_buffer, byte_fallback)
    }
}
