pub struct TrainerUtils;

impl TrainerUtils {
    #[inline(always)]
    pub fn byte_to_unicode_encode_fast<'a>(bytes: &[u8], out_buf: &'a mut [u8; 512]) -> &'a [u8] {
        let mut write_idx = 0;

        for &b in bytes {
            let u = b as u32;
            let c = match b {
                0..=32 | 127..=159 => u,
                33..=126 => u,
                160..=191 => u + 0x0100 - 33,
                192..=222 => u + 0x0120 - 127,
                223..=255 => u + 0x0180 - 223,
            };

            unsafe {
                if c < 0x80 {
                    *out_buf.get_unchecked_mut(write_idx) = c as u8;
                    write_idx += 1;
                } else if c < 0x800 {
                    *out_buf.get_unchecked_mut(write_idx) = (0xC0 | (c >> 6)) as u8;
                    *out_buf.get_unchecked_mut(write_idx + 1) = (0x80 | (c & 0x3F)) as u8;
                    write_idx += 2;
                } else {
                    *out_buf.get_unchecked_mut(write_idx) = (0xE0 | (c >> 12)) as u8;
                    *out_buf.get_unchecked_mut(write_idx + 1) = (0x80 | ((c >> 6) & 0x3F)) as u8;
                    *out_buf.get_unchecked_mut(write_idx + 2) = (0x80 | (c & 0x3F)) as u8;
                    write_idx += 3;
                }
            }
        }

        unsafe { out_buf.get_unchecked(..write_idx) }
    }
}
