use super::simd_scaner_avx2::SimdScanner;

pub struct SimdPath;

impl SimdPath {
    #[inline(always)]
    pub fn try_scan_ascii(bytes_ptr: *const u8, len: usize, curr_idx: usize) -> Option<bool> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                if curr_idx + 32 <= len {
                    unsafe {
                        if SimdScanner::is_pure_ascii_chunk(bytes_ptr.add(curr_idx))
                            && SimdScanner::has_no_spaces(bytes_ptr.add(curr_idx))
                        {
                            return Some(true);
                        }
                    }
                }
            }
        }
        None
    }
}
