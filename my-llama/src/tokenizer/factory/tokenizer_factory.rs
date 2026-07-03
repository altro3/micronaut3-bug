use crate::tokenizer::factory::table::InlineHashTable;
use crate::tokenizer::factory::FactoryUtils;
use crate::tokenizer::BpeTokenizer;
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
        let raw_pairs = Vec::with_capacity(512000);
        let mut text_to_id_table = InlineHashTable::new();
        let mut vocab_size = 0;

        // ДОБАВЛЕНО: Временный буфер для упорядоченной компиляции обратного словаря (ID -> Токен)
        // Выделяем с запасом под размер вокабуляра Qwen (обычно ~152 000, ставим 260 000 для защиты)
        let mut vocab_compiled_tokens = vec![Vec::new(); 260000];

        unsafe {
            let base = file_ptr;
            let end_ptr = base.add(file_len);
            let mut cursor = base;

            // --- ЭТАП 1: ПАРСИНГ VOCAB ---
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
                            let raw_slice = std::slice::from_raw_parts(start_tok, (end_tok as usize) - (start_tok as usize));
                            let id_slice = std::slice::from_raw_parts(start_id, (end_id as usize) - (start_id as usize));
                            let id = FactoryUtils::parse_u32(id_slice);

                            let mut clean_tok = [0u8; 256];
                            let mut clen = 0;
                            let mut si = 0;
                            while si < raw_slice.len() {
                                if raw_slice[si] == b'\\' && si + 1 < raw_slice.len() {
                                    clean_tok[clen] = match raw_slice[si + 1] {
                                        b'"' => b'"',
                                        b'\\' => b'\\',
                                        b'n' => b'\n',
                                        b't' => b'\t',
                                        b'r' => b'\r',
                                        other => other,
                                    };
                                    si += 2;
                                } else {
                                    clean_tok[clen] = raw_slice[si];
                                    si += 1;
                                }
                                clen += 1;
                            }

                            // ЗАЩИЩЕННОЕ занесение в byte_fallback
                            if clen == 1 {
                                let byte_idx = *clean_tok.get_unchecked(0) as usize;
                                if *byte_fallback.get_unchecked(byte_idx) == 0 {
                                    *byte_fallback.get_unchecked_mut(byte_idx) = id;
                                }
                            }

                            let mut char_buf = [0u8; 4];
                            let decoded_len = FactoryUtils::decode_inplace(&clean_tok[..clen], &mut char_buf);
                            if decoded_len == 1 && clen > 1 {
                                let byte_idx = *char_buf.get_unchecked(0) as usize;
                                if *byte_fallback.get_unchecked(byte_idx) == 0 {
                                    *byte_fallback.get_unchecked_mut(byte_idx) = id;
                                }
                            }

                            // ДОБАВЛЕНО: Сохраняем очищенные байты токена в упорядоченный вектор обратного словаря рантайма
                            let id_idx = id as usize;
                            if id_idx >= vocab_compiled_tokens.len() {
                                vocab_compiled_tokens.resize(id_idx + 65536, Vec::new());
                            }
                            *vocab_compiled_tokens.get_unchecked_mut(id_idx) = clean_tok[..clen].to_vec();

                            let h = FactoryUtils::fxhash64(&clean_tok[..clen]);
                            text_to_id_table.insert(h, id);
                            vocab_size += 1;
                        }
                    } else {
                        cursor = cursor.add(1);
                    }
                }
            }

            // Усекаем вектор строго до максимального задействованного ID, убирая хвосты
            vocab_compiled_tokens.truncate(vocab_size.max(256));

            for b in 0..=255 {
                if *byte_fallback.get_unchecked(b) == 0 {
                    *byte_fallback.get_unchecked_mut(b) = b as u32;
                }
            }
            // --- ЭТАП 3: EOS ID SCAN С ЖЕСТКИМ СБРОСОМ КУРСОРA И АППАРАТНЫМ SIMD MATCH ---
            cursor = base; // ГАРАНТИРОВАННЫЙ СБРОС
            let mut eos_id = 248044; // Дефолтный фоллбэк Qwen
            let mut added_found = false;

            // Добиваем строку `"added_tokens":` (15 байт) до ровного 16-байтового регистра нулем
            let target_16 = *b"\"added_tokens\":\0";
            let target_vec = std::arch::x86_64::_mm_loadu_si128(target_16.as_ptr() as *const std::arch::x86_64::__m128i);

            while cursor.add(16) < end_ptr {
                // Загружаем 16 байт из файла в SSE-регистр за 1 такт CPU
                let file_vec = std::arch::x86_64::_mm_loadu_si128(cursor as *const std::arch::x86_64::__m128i);

                // Параллельно сравниваем все 16 байт одной инструкцией процессора
                let cmp = std::arch::x86_64::_mm_cmpeq_epi8(file_vec, target_vec);
                let mask = std::arch::x86_64::_mm_movemask_epi8(cmp) as u32;

                // Нам нужно, чтобы совпали первые 15 байт (выставляем битовую маску 0x7FFF)
                if (mask & 0x7FFF) == 0x7FFF {
                    cursor = cursor.add(15);
                    added_found = true;
                    break;
                }
                cursor = cursor.add(1);
            }

            if added_found {
                let mut off_ptr = cursor;

                // Аналогично готовим 16-байтовую маску для токена `<|endoftext|>` (13 байт) + паддинг
                let eos_16 = *b"<|endoftext|>\0\0\0";
                let eos_vec = std::arch::x86_64::_mm_loadu_si128(eos_16.as_ptr() as *const std::arch::x86_64::__m128i);

                while off_ptr.add(16) < end_ptr {
                    let file_vec = std::arch::x86_64::_mm_loadu_si128(off_ptr as *const std::arch::x86_64::__m128i);
                    let cmp = std::arch::x86_64::_mm_cmpeq_epi8(file_vec, eos_vec);
                    let mask = std::arch::x86_64::_mm_movemask_epi8(cmp) as u32;

                    // Нам нужно совпадение первых 13 байт (битовая маска 0x1FFF)
                    if (mask & 0x1FFF) == 0x1FFF {
                        let mut s = off_ptr;
                        let scan_limit = s.add(200);
                        while s < end_ptr && s < scan_limit {
                            // Проверяем ключ `"id"` (4 байта) через прямое чтение u32 из памяти за 1 такт
                            if std::ptr::read(s as *const u32) == u32::from_le_bytes(*b"\"id\"") {
                                s = s.add(4);
                                while s < end_ptr && !(*s).is_ascii_digit() {
                                    s = s.add(1);
                                }
                                let start_d = s;
                                while s < end_ptr && (*s).is_ascii_digit() {
                                    s = s.add(1);
                                }

                                let len_d = (s as usize) - (start_d as usize);
                                let id_slice = std::slice::from_raw_parts(start_d, len_d);
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

            Ok(BpeTokenizer::new(&raw_pairs, byte_fallback, eos_id, vocab_size, &vocab_compiled_tokens))
        }
    }
}
