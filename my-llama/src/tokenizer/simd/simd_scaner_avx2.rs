use std::arch::x86_64::*;

pub struct SimdScanner;

impl SimdScanner {
    #[target_feature(enable = "avx2")]
    pub unsafe fn is_pure_ascii_chunk(bytes_ptr: *const u8) -> bool {
        let chunk = unsafe { _mm256_loadu_si256(bytes_ptr as *const __m256i) };
        let v_high_bit_mask = _mm256_set1_epi8(i8::MIN);
        let mask = _mm256_movemask_epi8(_mm256_and_si256(chunk, v_high_bit_mask)) as u32;
        mask == 0
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn has_no_spaces(bytes_ptr: *const u8) -> bool {
        let chunk = unsafe { _mm256_loadu_si256(bytes_ptr as *const __m256i) };
        let v_space = _mm256_set1_epi8(32);

        let m_space = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_space)) as u32;
        m_space == 0
    }
}
