pub struct TrainerUtils;

impl TrainerUtils {
    #[inline(always)]
    pub fn bytes_to_qwen_string(bytes: &[u8]) -> String {
        let mut result = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            match b {
                0x20 => result.push('Ġ'),
                0x0A => result.push('Ċ'),
                0x0D => result.push('ĉ'),
                0x09 => result.push('ĉ'),
                _ => {
                    result.push(b as char);
                }
            }
        }
        result
    }
}
