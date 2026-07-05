use std::collections::HashMap;

pub struct TrainerUtils;

fn bytes_to_unicode_mapping() -> Vec<char> {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);

    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n = 0;

    for b in 0..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    let mut mapping = vec![' '; 256];
    for i in 0..bs.len() {
        mapping[bs[i] as usize] = cs[i];
    }
    mapping
}

thread_local! {
    static ENCODE_MAP: Vec<char> = bytes_to_unicode_mapping();
    static DECODE_MAP: HashMap<char, u8> = {
        let map = bytes_to_unicode_mapping();
        map.into_iter().enumerate().map(|(b, c)| (c, b as u8)).collect()
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
