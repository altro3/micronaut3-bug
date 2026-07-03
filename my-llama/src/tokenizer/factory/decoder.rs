pub struct FactoryUtils;

impl FactoryUtils {
    #[inline(always)]
    pub fn fxhash64(b: &[u8]) -> u64 {
        let (mut hash, mut c, len) = (0u64, 0, b.len());
        while c + 8 <= len {
            let mut block = 0u64;
            unsafe {
                std::ptr::copy_nonoverlapping(b.as_ptr().add(c), &mut block as *mut u64 as *mut u8, 8);
            }
            hash = hash.rotate_left(5) ^ block;
            hash = hash.wrapping_mul(0x517cc1b727220a95);
            c += 8;
        }
        if c < len {
            let mut block = 0u64;
            unsafe {
                std::ptr::copy_nonoverlapping(b.as_ptr().add(c), &mut block as *mut u64 as *mut u8, len - c);
            }
            hash = hash.rotate_left(5) ^ block;
            hash = hash.wrapping_mul(0x517cc1b727220a95);
        }
        hash
    }

    #[inline(always)]
    pub fn decode_inplace(s: &[u8], buf: &mut [u8; 128]) -> usize {
        let (mut w, mut i, len) = (0, 0, s.len());
        while i < len && w < 128 {
            let b0 = s[i];
            let u = if b0 < 0x80 {
                i += 1;
                b0 as u32
            } else if (b0 & 0xE0) == 0xC0 && i + 1 < len {
                let u = (((b0 & 0x1F) as u32) << 6) | ((s[i + 1] & 0x3F) as u32);
                i += 2;
                u
            } else if (b0 & 0xF0) == 0xE0 && i + 2 < len {
                let u = (((b0 & 0x0F) as u32) << 12) | (((s[i + 1] & 0x3F) as u32) << 6) | ((s[i + 2] & 0x3F) as u32);
                i += 3;
                u
            } else {
                i += 1;
                continue;
            };

            buf[w] = match u {
                0..=32 | 127..=159 => u as u8,
                33..=126 => u as u8,
                0x0100..=0x011F => (u - 0x0100 + 33) as u8,
                0x0120..=0x013E => (u - 0x0120 + 127) as u8,
                0x013F..=0x015F => (u - 0x0180 + 223) as u8,
                _ => u as u8,
            };
            w += 1;
        }
        w
    }

    #[inline(always)]
    pub fn parse_u32(bytes: &[u8]) -> u32 {
        bytes
            .iter()
            .fold(0u32, |acc, &b| if b.is_ascii_digit() { acc * 10 + (b - b'0') as u32 } else { acc })
    }
}
