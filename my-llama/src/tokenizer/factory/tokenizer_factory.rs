use crate::tokenizer::factory::table::InlineHashTable;
use crate::tokenizer::factory::FactoryUtils;
use crate::tokenizer::BpeTokenizer;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Read};

pub struct TokenizerFactory;

impl TokenizerFactory {
    pub fn from_file(file_path: &str) -> std::io::Result<BpeTokenizer> {
        let file = File::open(file_path)?;
        let mut file_buf = Vec::with_capacity(file.metadata()?.len() as usize);
        BufReader::new(file).read_to_end(&mut file_buf)?;
        let file_len = file_buf.len();

        let (mut byte_fallback, mut raw_pairs) = ([0u32; 256], Vec::with_capacity(256000));
        let mut hash_table = InlineHashTable::new();
        let (mut tmp_buf, mut vocab_size) = ([0u8; 128], 0);

        if let Some(v_start) = file_buf.windows(7).position(|w| w == b"\"vocab\"") {
            let mut cursor = v_start + 7;
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

        // --- ИСПРАВЛЕННЫЙ ЭТАП 2: MERGES SCAN ЧЕРЕЗ ПРЯМЫЕ ID ---
        if let Some(m_start) = file_buf.windows(8).position(|w| w == b"\"merges\"") {
            let mut cursor = m_start + 8;
            while cursor < file_len && file_buf[cursor] != b'[' {
                cursor += 1;
            }
            cursor += 1;
            let mut rank = 0u32;

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
                        // Прямой парсинг ID чисел из строки "ID1 ID2" без декодеров и хэшей!
                        let id1 = FactoryUtils::parse_u32(&m_line[..sp]);
                        let id2 = FactoryUtils::parse_u32(&m_line[sp + 1..]);

                        if id1 != u32::MAX && id2 != u32::MAX {
                            // Нам нужен результирующий ID склеенного токена.
                            // Но мы знаем, что мёрджи идут строго по порядку их генерации тренером (current_id = 256 + rank)
                            let mid = 256 + rank;
                            raw_pairs.push((((id1 as u64) << 32) | (id2 as u64), (rank, mid)));
                            rank += 1;
                        }
                    }
                } else {
                    cursor += 1;
                }
            }
        }

        let mut eos_id = 248044;
        if let Some(a_start) = file_buf.windows(13).position(|w| w == b"\"added_tokens\"") {
            if let Some(off) = file_buf[a_start + 13..].windows(13).position(|w| w == b"<|endoftext|>") {
                let mut s = a_start + 13 + off;
                while s < file_len && s < a_start + 213 + off {
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
