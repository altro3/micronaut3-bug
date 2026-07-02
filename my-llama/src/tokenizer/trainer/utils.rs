pub struct TrainerUtils;

const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

impl TrainerUtils {
    #[inline(always)]
    pub fn byte_to_unicode_encode_fast<'a>(bytes: &[u8], out_buf: &'a mut [u8; 512]) -> &'a [u8] {
        let mut write_idx = 0;

        for &b in bytes {
            unsafe {
                *out_buf.get_unchecked_mut(write_idx) = *HEX_DIGITS.get_unchecked((b >> 4) as usize);
                *out_buf.get_unchecked_mut(write_idx + 1) = *HEX_DIGITS.get_unchecked((b & 0x0F) as usize);
            }
            write_idx += 2;
        }
        unsafe { out_buf.get_unchecked(..write_idx) }
    }
}
