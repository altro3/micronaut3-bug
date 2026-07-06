pub struct HfByteDecoder;

impl HfByteDecoder {
    pub fn decode_string(s: &str) -> Vec<u8> {
        let mut raw_bytes = Vec::with_capacity(s.len());

        for c in s.chars() {
            let cp = c as u32;
            let byte = match cp {
                0x00..=0x7F => cp as u8,
                0x00A0..=0x00FF => cp as u8,
                0x0100..=0x011F => (cp - 0x0100) as u8,
                0x0120..=0x013F => (cp - 0x0120 + 127) as u8,
                0x0140 => 173,
                _ => {
                    let mut buf = [0; 4];
                    let utf8_str = c.encode_utf8(&mut buf);
                    raw_bytes.extend_from_slice(utf8_str.as_bytes());
                    continue;
                },
            };
            raw_bytes.push(byte);
        }

        raw_bytes
    }
}
