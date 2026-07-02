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

    /// Конвертирует Hex-строку из JSON обратно в сырые байты без аллокаций.
    #[inline(always)]
    pub fn decode_inplace(s: &[u8], buf: &mut [u8; 128]) -> usize {
        let mut w = 0;
        let mut i = 0;
        let len = s.len();

        while i + 1 < len && w < 128 {
            let h1 = match s[i] {
                b'0'..=b'9' => s[i] - b'0',
                b'A'..=b'F' => s[i] - b'A' + 10,
                _ => {
                    i += 1;
                    continue;
                }
            };
            let h2 = match s[i + 1] {
                b'0'..=b'9' => s[i + 1] - b'0',
                b'A'..=b'F' => s[i + 1] - b'A' + 10,
                _ => {
                    i += 2;
                    continue;
                }
            };
            buf[w] = (h1 << 4) | h2;
            i += 2;
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
