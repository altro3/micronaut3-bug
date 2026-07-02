use crate::tokenizer::factory::table::InlineHashTable;
use crate::tokenizer::factory::FactoryUtils;
use crate::tokenizer::BpeTokenizer;
use std::arch::x86_64::*;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Read};

pub struct TokenizerFactory;

impl TokenizerFactory {
    #[target_feature(enable = "avx2")]
    fn find_sub_simd(buf: &[u8], needle: &[u8]) -> Option<usize> {
        if buf.len() < needle.len() {
            return None;
        }
        let n0 = _mm256_set1_epi8(needle[0] as i8);
        let limit = buf.len() - needle.len();
        let mut i = 0;
        while i + 32 <= limit {
            let chunk = unsafe { _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i) };
            let cmp = _mm256_cmpeq_epi8(chunk, n0);
            let mut mask = _mm256_movemask_epi8(cmp) as u32;
            while mask != 0 {
                let tz = mask.trailing_zeros() as usize;
                if &buf[i + tz..i + tz + needle.len()] == needle {
                    return Some(i + tz);
                }
                mask &= mask - 1;
            }
            i += 32;
        }
        buf[i..].windows(needle.len()).position(|w| w == needle).map(|pos| i + pos)
    }

    pub fn from_file(file_path: &str) -> std::io::Result<BpeTokenizer> {
        let file = File::open(file_path)?;
        let mut file_buf = Vec::with_capacity(file.metadata()?.len() as usize);
        BufReader::new(file).read_to_end(&mut file_buf)?;
        let file_len = file_buf.len();

        let (mut byte_fallback, mut raw_pairs) = ([0u32; 256], Vec::with_capacity(256000));
        let mut hash_table = InlineHashTable::new();
        let (mut tmp_buf, mut vocab_size) = ([0u8; 128], 0);

        let v_start = unsafe { Self::find_sub_simd(&file_buf, b"\"vocab\"") };
        if let Some(start) = v_start {
            let mut cursor = start + 7;
            while cursor < file_len && file_buf[cursor] != b'{' {
                cursor += 1;
            }
            cursor += 1;
            let mut depth = 1;
            while cursor < file_len && depth > 0 {
                if file_buf[cursor] == b'}' {
                    depth -= 1;
                    cursor += 1;
                    continue;
                }
                if file_buf[cursor] == b'{' {
                    depth += 1;
                    cursor += 1;
                    continue;
                }
                if file_buf[cursor] == b'"' {
                    let start_tok = cursor + 1;
                    cursor += 1;
                    while cursor < file_len && file_buf[cursor] != b'"' {
                        cursor += if file_buf[cursor] == b'\\' { 2 } else { 1 };
                    }
                    let end_tok = cursor;
                    cursor += 1;
                    while cursor < file_len && file_buf[cursor] != b':' {
                        cursor += 1;
                    }
                    cursor += 1;
                    while cursor < file_len && file_buf[cursor].is_ascii_whitespace() {
                        cursor += 1;
                    }
                    let start_id = cursor;
                    while cursor < file_len && file_buf[cursor].is_ascii_digit() {
                        cursor += 1;
                    }

                    if start_tok < end_tok && start_id < cursor {
                        let id = FactoryUtils::parse_u32(&file_buf[start_id..cursor]);
                        let len = FactoryUtils::decode_inplace(&file_buf[start_tok..end_tok], &mut tmp_buf);
                        if len == 1 {
                            byte_fallback[tmp_buf[0] as usize] = id;
                        }
                        let h = FactoryUtils::fxhash64(&tmp_buf[..len]);
                        hash_table.insert(h, id);
                        vocab_size += 1;
                    }
                } else {
                    cursor += 1;
                }
            }
        }
        for b in 0..=255 {
            if byte_fallback[b] == 0 {
                byte_fallback[b] = b as u32;
            }
        }

        let m_start = unsafe { Self::find_sub_simd(&file_buf, b"\"merges\"") };
        if let Some(start) = m_start {
            let mut cursor = start + 8;
            while cursor < file_len && file_buf[cursor] != b'[' {
                cursor += 1;
            }
            cursor += 1;
            let (mut rank, mut p1_b, mut p2_b, mut m_b) = (0u32, [0u8; 128], [0u8; 128], [0u8; 256]);
            while cursor < file_len && file_buf[cursor] != b']' {
                if file_buf[cursor] == b'"' {
                    let start_str = cursor + 1;
                    cursor += 1;
                    while cursor < file_len && file_buf[cursor] != b'"' {
                        cursor += 1;
                    }
                    let m_line = &file_buf[start_str..cursor];
                    cursor += 1;
                    if let Some(sp) = m_line.iter().position(|&b| b == b' ') {
                        let p1_len = FactoryUtils::decode_inplace(&m_line[..sp], &mut p1_b);
                        let p2_len = FactoryUtils::decode_inplace(&m_line[sp + 1..], &mut p2_b);
                        let id1 = if p1_len == 1 {
                            byte_fallback[p1_b[0] as usize]
                        } else {
                            hash_table.find(FactoryUtils::fxhash64(&p1_b[..p1_len]))
                        };
                        let id2 = if p2_len == 1 {
                            byte_fallback[p2_b[0] as usize]
                        } else {
                            hash_table.find(FactoryUtils::fxhash64(&p2_b[..p2_len]))
                        };

                        if id1 != u32::MAX && id2 != u32::MAX {
                            unsafe {
                                std::ptr::copy_nonoverlapping(p1_b.as_ptr(), m_b.as_mut_ptr(), p1_len);
                                std::ptr::copy_nonoverlapping(p2_b.as_ptr(), m_b.as_mut_ptr().add(p1_len), p2_len);
                            }
                            let mid = hash_table.find(FactoryUtils::fxhash64(&m_b[..p1_len + p2_len]));
                            let final_mid = if mid == u32::MAX { rank } else { mid };
                            raw_pairs.push((((id1 as u64) << 32) | (id2 as u64), (rank, final_mid)));
                            rank += 1;
                        }
                    }
                } else {
                    cursor += 1;
                }
            }
        }

        let a_start = unsafe { Self::find_sub_simd(&file_buf, b"\"added_tokens\"") };
        let mut eos_id = 248044;
        if let Some(start) = a_start {
            if let Some(off) = file_buf[start + 13..].windows(13).position(|w| w == b"<|endoftext|>") {
                let mut s = start + 13 + off;
                while s < file_len && s < start + 213 + off {
                    if file_buf[s..].starts_with(b"\"id\"") {
                        s += 4;
                        while s < file_len && !file_buf[s].is_ascii_digit() {
                            s += 1;
                        }
                        let start_d = s;
                        while s < file_len && file_buf[s].is_ascii_digit() {
                            s += 1;
                        }
                        eos_id = FactoryUtils::parse_u32(&file_buf[start_d..s]);
                        break;
                    }
                    s += 1;
                }
            }
        }
        if vocab_size == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "Malformed JSON"));
        }
        Ok(BpeTokenizer::new(&raw_pairs, byte_fallback, eos_id, vocab_size))
    }
}
