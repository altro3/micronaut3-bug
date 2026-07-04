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
                        if SimdScanner::is_pure_ascii_chunk(chunk_ptr(bytes_ptr, curr_idx))
                            && SimdScanner::has_no_spaces(chunk_ptr(bytes_ptr, curr_idx))
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

#[inline(always)]
unsafe fn chunk_ptr(base: *const u8, offset: usize) -> *const u8 {
    base.add(offset)
}
