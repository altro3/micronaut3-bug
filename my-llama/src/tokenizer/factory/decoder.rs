pub struct FactoryUtils;

impl FactoryUtils {
    #[inline(always)]
    pub fn fxhash64(b: &[u8]) -> u64 {
        let len = b.len();
        let mut hash = 0u64;
        let mut c = 0;
        let ptr = b.as_ptr();

        while c + 8 <= len {
            let block = unsafe { std::ptr::read_unaligned(ptr.add(c) as *const u64) };
            hash = hash.rotate_left(5) ^ block;
            hash = hash.wrapping_mul(0x517cc1b727220a95);
            c += 8;
        }

        if c < len {
            let mut block = 0u64;
            let mut shift = 0;
            while c < len {
                block |= (unsafe { *ptr.add(c) } as u64) << shift;
                shift += 8;
                c += 1;
            }
            hash = hash.rotate_left(5) ^ block;
            hash = hash.wrapping_mul(0x517cc1b727220a95);
        }
        hash
    }

    #[inline(always)]
    pub fn decode_inplace(s: &[u8], buf: &mut [u8]) -> usize {
        let len = s.len();
        let buf_len = buf.len();
        let s_ptr = s.as_ptr();
        let buf_ptr = buf.as_mut_ptr();

        let (mut i, mut w) = (0, 0);

        unsafe {
            while i < len && w < buf_len {
                let b0 = *s_ptr.add(i);
                let mut cp: u32;

                if b0 < 0x80 {
                    cp = b0 as u32;
                    i += 1;
                } else if (b0 & 0xE0) == 0xC0 && i + 1 < len {
                    cp = ((b0 & 0x1F) as u32) << 6;
                    cp |= (*s_ptr.add(i + 1) & 0x3F) as u32;
                    i += 2;
                } else if (b0 & 0xF0) == 0xE0 && i + 2 < len {
                    cp = ((b0 & 0x0F) as u32) << 12;
                    cp |= ((*s_ptr.add(i + 1) & 0x3F) as u32) << 6;
                    cp |= (*s_ptr.add(i + 2) & 0x3F) as u32;
                    i += 3;
                } else if (b0 & 0xF8) == 0xF0 && i + 3 < len {
                    cp = ((b0 & 0x07) as u32) << 18;
                    cp |= ((*s_ptr.add(i + 1) & 0x3F) as u32) << 12;
                    cp |= ((*s_ptr.add(i + 2) & 0x3F) as u32) << 6;
                    cp |= (*s_ptr.add(i + 3) & 0x3F) as u32;
                    i += 4;
                } else {
                    i += 1;
                    continue;
                }

                let raw_byte = match cp {
                    0x00..=0x7F => cp as u8,
                    0x0100..=0x0120 => (cp - 0x0100) as u8,
                    0x0121..=0x017D => (cp - 0x0121 + 33) as u8,
                    0x017E..=0x01AC => (cp - 0x017E + 127) as u8,
                    0x01AD..=0x01FF => (cp - 0x01AD + 174) as u8,
                    _ => cp as u8,
                };

                *buf_ptr.add(w) = raw_byte;
                w += 1;
            }
        }
        w
    }

    #[inline(always)]
    pub fn parse_u32(bytes: &[u8]) -> u32 {
        let mut val = 0u32;
        let len = bytes.len();
        let ptr = bytes.as_ptr();

        for i in 0..len {
            unsafe {
                let digit = *ptr.add(i) - b'0';
                val = val * 10 + digit as u32;
            }
        }
        val
    }
}
