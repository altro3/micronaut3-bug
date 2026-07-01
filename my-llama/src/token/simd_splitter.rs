#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct SimdSplitter;

impl SimdSplitter {
    #[inline(always)]
    pub fn split(text: &str, mut f: impl for<'b> FnMut(&'b [u8])) {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut start = 0;

        if len >= 32 && is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                let v_space1 = _mm256_set1_epi8(b' ' as i16 as i8);
                let v_space2 = _mm256_set1_epi8(b'\n' as i16 as i8);
                let v_space3 = _mm256_set1_epi8(b'\r' as i16 as i8);
                let v_space4 = _mm256_set1_epi8(b'\t' as i16 as i8);

                while i <= len - 32 {
                    let ptr = bytes.as_ptr().add(i);
                    let chunk = _mm256_loadu_si256(ptr as *const __m256i);

                    let cmp1 = _mm256_cmpeq_epi8(chunk, v_space1);
                    let cmp2 = _mm256_cmpeq_epi8(chunk, v_space2);
                    let cmp3 = _mm256_cmpeq_epi8(chunk, v_space3);
                    let cmp4 = _mm256_cmpeq_epi8(chunk, v_space4);

                    let or1 = _mm256_or_si256(cmp1, cmp2);
                    let or2 = _mm256_or_si256(cmp3, cmp4);
                    let final_mask = _mm256_or_si256(or1, or2);

                    let bitmask = _mm256_movemask_epi8(final_mask) as u32;

                    if bitmask != 0 {
                        let first_space_idx = bitmask.trailing_zeros() as usize;
                        let split_pos = i + first_space_idx;

                        if split_pos > start {
                            f(&bytes[start..split_pos]);
                        }
                        start = split_pos + 1;
                        i = start;
                        continue;
                    }
                    i += 32;
                }
            }
        }

        for i in start..len {
            let b = unsafe { *bytes.get_unchecked(i) };
            if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                if i > start {
                    f(&bytes[start..i]);
                }
                start = i + 1;
            }
        }

        if start < len {
            f(&bytes[start..len]);
        }
    }
}
