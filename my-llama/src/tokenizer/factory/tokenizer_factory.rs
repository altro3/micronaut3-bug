use crate::tokenizer::factory::table::InlineHashTable;
use crate::tokenizer::factory::FactoryUtils;
use crate::tokenizer::BpeTokenizer;
use std::arch::x86_64::{__m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};
use std::fs::File;
use std::io::{Error, ErrorKind, Read};

pub struct TokenizerFactory;

impl TokenizerFactory {
    pub fn from_file(file_path: &str) -> std::io::Result<BpeTokenizer> {
        let mut file = File::open(file_path)?;
        let file_len = file.metadata()?.len() as usize;

        let layout = std::alloc::Layout::from_size_align(file_len, 64).unwrap();
        let file_ptr = unsafe {
            let p = std::alloc::alloc(layout);
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            p
        };

        unsafe {
            let slice = std::slice::from_raw_parts_mut(file_ptr, file_len);
            file.read_exact(slice)?;
        }

        let mut byte_fallback = [0u32; 256];
        let mut raw_pairs = Vec::with_capacity(512000);
        let mut hash_table = InlineHashTable::new();
        let mut tmp_buf = [0u8; 256];
        let mut vocab_size = 0;

        unsafe {
            let base = file_ptr;
            let end_ptr = base.add(file_len);
            let mut cursor = base;

            let mut vocab_found = false;
            while cursor.add(7) < end_ptr {
                if std::ptr::read(cursor as *const [u8; 7]) == *b"\"vocab\"" {
                    cursor = cursor.add(7);
                    vocab_found = true;
                    break;
                }
                cursor = cursor.add(1);
            }

            if vocab_found {
                while cursor < end_ptr && *cursor != b'{' {
                    cursor = cursor.add(1);
                }
                cursor = cursor.add(1);

                let mut depth = 1;
                while cursor < end_ptr && depth > 0 {
                    let b = *cursor;
                    if b == b'}' {
                        depth -= 1;
                        cursor = cursor.add(1);
                        continue;
                    }
                    if b == b'{' {
                        depth += 1;
                        cursor = cursor.add(1);
                        continue;
                    }

                    if b == b'"' {
                        let start_tok = cursor.add(1);
                        cursor = cursor.add(1);
                        while cursor < end_ptr && *cursor != b'"' {
                            cursor = cursor.add(if *cursor == b'\\' { 2 } else { 1 });
                        }
                        let end_tok = cursor;
                        cursor = cursor.add(1);

                        while cursor < end_ptr && *cursor != b':' {
                            cursor = cursor.add(1);
                        }
                        cursor = cursor.add(1);
                        while cursor < end_ptr && (*cursor).is_ascii_whitespace() {
                            cursor = cursor.add(1);
                        }

                        let start_id = cursor;
                        while cursor < end_ptr && (*cursor).is_ascii_digit() {
                            cursor = cursor.add(1);
                        }
                        let end_id = cursor;

                        if start_tok < end_tok && start_id < end_id {
                            let tok_slice = std::slice::from_raw_parts(start_tok, end_tok_offset(start_tok, end_tok));
                            let id_slice = std::slice::from_raw_parts(start_id, end_tok_offset(start_id, end_id));

                            let id = FactoryUtils::parse_u32(id_slice);
                            let len = FactoryUtils::decode_inplace(tok_slice, &mut tmp_buf);

                            if len == 1 {
                                *byte_fallback.get_unchecked_mut(*tmp_buf.get_unchecked(0) as usize) = id;
                            }

                            let h = FactoryUtils::fxhash64(&tmp_buf[..len]);
                            hash_table.insert(h, id);
                            vocab_size += 1;
                        }
                    } else {
                        cursor = cursor.add(1);
                    }
                }
            }

            for b in 0..=255 {
                if *byte_fallback.get_unchecked(b) == 0 {
                    *byte_fallback.get_unchecked_mut(b) = b as u32;
                }
            }

            cursor = base;
            let mut merges_found = false;
            while cursor.add(8) < end_ptr {
                if std::ptr::read(cursor as *const [u8; 8]) == *b"\"merges\"" {
                    cursor = cursor.add(8);
                    merges_found = true;
                    break;
                }
                cursor = cursor.add(1);
            }

            if merges_found {
                while cursor < end_ptr && *cursor != b'[' {
                    cursor = cursor.add(1);
                }
                cursor = cursor.add(1);
                let mut rank = 0u32;

                while cursor < end_ptr && *cursor != b']' {
                    if *cursor == b'"' {
                        let start_str = cursor.add(1);
                        cursor = cursor.add(1);
                        while cursor < end_ptr && *cursor != b'"' {
                            cursor = cursor.add(1);
                        }
                        let end_str = cursor;
                        cursor = cursor.add(1);

                        let mut p = start_str;
                        while p < end_str && *p != b' ' {
                            p = p.add(1);
                        }

                        if p < end_str {
                            let left_raw = std::slice::from_raw_parts(start_str, end_tok_offset(start_str, p));
                            let left_len = FactoryUtils::decode_inplace(left_raw, &mut tmp_buf);
                            let left_hash = FactoryUtils::fxhash64(&tmp_buf[..left_len]);
                            let id1 = hash_table.find(left_hash);

                            let right_raw = std::slice::from_raw_parts(p.add(1), end_tok_offset(p.add(1), end_str));
                            let right_len = FactoryUtils::decode_inplace(right_raw, &mut tmp_buf);
                            let right_hash = FactoryUtils::fxhash64(&tmp_buf[..right_len]);
                            let id2 = hash_table.find(right_hash);

                            // Вычисляем результирующий скомпилированный ID склеенного токена
                            let mut full_buf = [0u8; 256];
                            let len_left = end_tok_offset(start_str, p);
                            let len_right = end_tok_offset(p.add(1), end_str);
                            std::ptr::copy_nonoverlapping(start_str, full_buf.as_mut_ptr(), len_left);
                            std::ptr::copy_nonoverlapping(p.add(1), full_buf.as_mut_ptr().add(len_left), len_right);

                            let full_raw = &full_buf[..len_left + len_right];
                            let full_len = FactoryUtils::decode_inplace(full_raw, &mut tmp_buf);
                            let full_hash = FactoryUtils::fxhash64(&tmp_buf[..full_len]);
                            let mid = hash_table.find(full_hash);

                            if id1 != u32::MAX && id2 != u32::MAX && mid != u32::MAX {
                                raw_pairs.push((((id1 as u64) << 32) | (id2 as u64), (rank, mid)));
                                rank += 1;
                            }
                        }
                    } else {
                        cursor = cursor.add(1);
                    }
                }
            }

            cursor = base;
            let mut eos_id = 248044;
            let mut added_found = false;

            // Готовим 16-байтную маску (захватываем двоеточие и кавычку, чтобы добить до 16 байт)
            // Строка: `"added_tokens":` (ровно 15 байт) + зануляем последний байт
            let target_16 = *b"\"added_tokens\":\0";
            let target_vec = _mm_loadu_si128(target_16.as_ptr() as *const __m128i);

            while cursor.add(16) < end_ptr {
                // Загружаем 16 байт из файла за 1 такт процессора
                let file_vec = _mm_loadu_si128(cursor as *const __m128i);

                // Сравниваем байты аппаратно
                let cmp = _mm_cmpeq_epi8(file_vec, target_vec);
                let mask = _mm_movemask_epi8(cmp) as u32;

                // Нам нужно, чтобы совпали первые 15 байт (битовая маска 0x7FFF)
                if (mask & 0x7FFF) == 0x7FFF {
                    cursor = cursor.add(15);
                    added_found = true;
                    break;
                }
                cursor = cursor.add(1);
            }

            if added_found {
                let mut off_ptr = cursor;

                let eos_16 = *b"<|endoftext|>\0\0\0";
                let eos_vec = _mm_loadu_si128(eos_16.as_ptr() as *const __m128i);

                while off_ptr.add(16) < end_ptr {
                    let file_vec = _mm_loadu_si128(off_ptr as *const __m128i);
                    let cmp = _mm_cmpeq_epi8(file_vec, eos_vec);
                    let mask = _mm_movemask_epi8(cmp) as u32;

                    // Нам нужно совпадение первых 13 байт (битовая маска 0x1FFF)
                    if (mask & 0x1FFF) == 0x1FFF {
                        let mut s = off_ptr;
                        let scan_limit = s.add(200);
                        while s < end_ptr && s < scan_limit {
                            // Проверяем `"id"` (4 байта) через прямое чтение u32 за 1 такт
                            if std::ptr::read(s as *const u32) == u32::from_le_bytes(*b"\"id\"") {
                                s = s.add(4);
                                while s < end_ptr && !(*s).is_ascii_digit() {
                                    s = s.add(1);
                                }
                                let start_d = s;
                                while s < end_ptr && (*s).is_ascii_digit() {
                                    s = s.add(1);
                                }

                                let id_slice = std::slice::from_raw_parts(start_d, end_tok_offset(start_d, s));
                                eos_id = FactoryUtils::parse_u32(id_slice);
                                break;
                            }
                            s = s.add(1);
                        }
                        break;
                    }
                    off_ptr = off_ptr.add(1);
                }
            }

            std::alloc::dealloc(file_ptr, layout);

            if vocab_size == 0 {
                return Err(Error::new(ErrorKind::InvalidData, "Malformed JSON"));
            }

            Ok(BpeTokenizer::new(&raw_pairs, byte_fallback, eos_id, vocab_size))
        }
    }
}

#[inline(always)]
fn end_tok_offset(start: *const u8, end: *const u8) -> usize {
    (end as usize) - (start as usize)
}
