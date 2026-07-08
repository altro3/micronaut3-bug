use fxhash::FxHashMap;

pub struct TrainerUtils;

fn bytes_to_unicode_mapping() -> Vec<(u8, char)> {
    let mut pairs = Vec::with_capacity(256);

    for b in 0..=255 {
        let b_u32 = b as u32;
        let cp = match b {
            0..=31 => b_u32 + 0x0100,
            32..=126 => b_u32,
            127..=158 => b_u32 - 127 + 0x0120,
            159..=255 => b_u32,
        };
        let final_cp = if b == 173 { 0x0140 } else { cp };
        pairs.push((b, char::from_u32(final_cp).unwrap()));
    }

    pairs
}

thread_local! {
    static ENCODE_MAP: Vec<char> = {
        let pairs = bytes_to_unicode_mapping();
        let mut map = vec!['\0'; 256];
        for (b, c) in pairs {
            map[b as usize] = c;
        }
        map
    };

    static DECODE_MAP: FxHashMap<char, u8> = {
        let pairs = bytes_to_unicode_mapping();
        let mut map = FxHashMap::with_capacity_and_hasher(256, Default::default());
        for (b, c) in pairs {
            map.insert(c, b);
        }
        map
    };
}

impl TrainerUtils {
    #[inline(always)]
    pub fn bytes_to_qwen_string(bytes: &[u8]) -> String {
        let mut result = String::with_capacity(bytes.len());
        ENCODE_MAP.with(|map| {
            for &b in bytes {
                result.push(map[b as usize]);
            }
        });
        result
    }

    #[inline(always)]
    pub fn qwen_string_to_bytes(qwen_str: &str) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(qwen_str.len());
        DECODE_MAP.with(|map| {
            for c in qwen_str.chars() {
                if let Some(&b) = map.get(&c) {
                    bytes.push(b);
                }
            }
        });
        bytes
    }
}
