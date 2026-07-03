use crate::tokenizer::factory::table::InlineHashTable;
use crate::tokenizer::factory::FactoryUtils;
use crate::tokenizer::BpeTokenizer;
use std::fs::File;
use std::io::{Error, ErrorKind, Read};

pub struct TokenizerFactory;

impl TokenizerFactory {
    pub fn from_file(file_path: &str) -> std::io::Result<Box<BpeTokenizer>> {
        let mut file = File::open(file_path)?;
        let file_len = file.metadata()?.len() as usize;

        let layout = std::alloc::Layout::from_size_align(file_len, 64).unwrap();
        let file_ptr = unsafe {
            let p = std::alloc::alloc(layout);
            if p.is_null() { std::alloc::handle_alloc_error(layout); }
            p
        };

        unsafe {
            let slice = std::slice::from_raw_parts_mut(file_ptr, file_len);
            file.read_exact(slice)?;
        }

        let mut byte_fallback = [0u32; 256];
        let mut raw_pairs = Vec::with_capacity(512000);
        let mut text_to_id_table = InlineHashTable::new();
        let mut vocab_size = 0;

        let mut vocab_compiled_tokens = Vec::with_capacity(260000);

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
                while cursor < end_ptr && *cursor != b'{' { cursor = cursor.add(1); }
                cursor = cursor.add(1);

                let mut depth = 1;
                while cursor < end_ptr && depth > 0 {
                    let b = *cursor;
                    if b == b'}' { depth -= 1; cursor = cursor.add(1); continue; }
                    if b == b'{' { depth += 1; cursor = cursor.add(1); continue; }

                    if b == b'"' {
                        let start_tok = cursor.add(1);
                        cursor = cursor.add(1);
                        while cursor < end_ptr && *cursor != b'"' {
                            cursor = cursor.add(if *cursor == b'\\' { 2 } else { 1 });
                        }
                        let end_tok = cursor;
                        cursor = cursor.add(1);

                        while cursor < end_ptr && *cursor != b':' { cursor = cursor.add(1); }
                        cursor = cursor.add(1);
                        while cursor < end_ptr && (*cursor).is_ascii_whitespace() { cursor = cursor.add(1); }

                        let start_id = cursor;
                        while cursor < end_ptr && (*cursor).is_ascii_digit() { cursor = cursor.add(1); }
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
                                        b'"' => b'"', b'\\' => b'\\', b'n' => b'\n', b't' => b'\t', b'r' => b'\r',
                                        other => other,
                                    };
                                    si += 2;
                                } else {
                                    clean_tok[clen] = raw_slice[si];
                                    si += 1;
                                }
                                clen += 1;
                            }

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

                            let id_idx = id as usize;
                            if id_idx >= vocab_compiled_tokens.len() {
                                vocab_compiled_tokens.resize(id_idx + 1, Vec::new());
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

            vocab_compiled_tokens.shrink_to_fit();
            // =================================================================
            // ВШИТЫЕ ЖЁСТКИЕ ЛОГИ ОРАКУЛА ДЛЯ АНАЛИЗА КИРИЛЛИЦЫ
            // =================================================================
            println!("\n[ОРАКУЛ-ФАБРИКА] Анализ byte_fallback перед заполнением дефолтами:");

            let id_d0 = byte_fallback[0xD0];
            println!("|-> Байт 0xD0 (208, старт русских букв) получил ID токена: {}", id_d0);
            if id_d0 != 0 && (id_d0 as usize) < vocab_compiled_tokens.len() {
                let token_bytes = &vocab_compiled_tokens[id_d0 as usize];
                println!("    |-> Из-за чего: имя токена в JSON: {:?}", token_bytes);
                println!("    |-> В виде строки: '{}'", String::from_utf8_lossy(token_bytes));
            } else {
                println!("    |-> Байт 0xD0 вообще НЕ ПОЛУЧИЛ ID (остался равен 0)!");
            }

            let id_b0 = byte_fallback[0xB0];
            println!("|-> Байт 0xB0 (176) получил ID токена: {}", id_b0);
            if id_b0 != 0 && (id_b0 as usize) < vocab_compiled_tokens.len() {
                let token_bytes = &vocab_compiled_tokens[id_b0 as usize];
                println!("    |-> Из-за чего: имя токена в JSON: {:?}", token_bytes);
                println!("    |-> В виде строки: '{}'", String::from_utf8_lossy(token_bytes));
            }

            println!("|-> Первые 5 непустых фоллбэков в не-ASCII диапазоне (128..255):");
            let mut printed = 0;
            for b_idx in 128..256 {
                let tid = byte_fallback[b_idx];
                if tid != 0 && printed < 5 {
                    let t_str = if (tid as usize) < vocab_compiled_tokens.len() {
                        String::from_utf8_lossy(&vocab_compiled_tokens[tid as usize]).to_string()
                    } else {
                        "UNKNOWN".to_string()
                    };
                    println!("    |-> Слот {}: забит токеном ID {} ('{}')", b_idx, tid, t_str);
                    printed += 1;
                }
            }
            println!("=================================================================\n");

            for b in 0..=255 {
                if *byte_fallback.get_unchecked(b) == 0 {
                    *byte_fallback.get_unchecked_mut(b) = b as u32;
                }
            }

            // =================================================================
            // --- ЭТАП 2: АППАРАТНЫЙ ПАРСЕР С ТОТАЛЬНЫМ ОТЛАДОЧНЫМ ОРАКУЛОМ ---
            // =================================================================
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
                while cursor < end_ptr && *cursor != b'[' { cursor = cursor.add(1); }
                cursor = cursor.add(1);
                let mut rank = 0u32;

                while cursor < end_ptr {
                    while cursor < end_ptr && *cursor != b'[' && *cursor != b']' {
                        cursor = cursor.add(1);
                    }
                    if cursor >= end_ptr { break; }

                    let b = *cursor;
                    if b == b']' { break; }

                    if b == b'[' {
                        cursor = cursor.add(1);

                        while cursor < end_ptr && *cursor != b'"' { cursor = cursor.add(1); }
                        if cursor >= end_ptr { break; }
                        let start_left = cursor.add(1);
                        cursor = cursor.add(1);
                        while cursor < end_ptr && *cursor != b'"' {
                            cursor = cursor.add(if *cursor == b'\\' { 2 } else { 1 });
                        }
                        let end_left = cursor;
                        cursor = cursor.add(1);

                        while cursor < end_ptr && *cursor != b'"' { cursor = cursor.add(1); }
                        if cursor >= end_ptr { break; }
                        let start_right = cursor.add(1);
                        cursor = cursor.add(1);
                        while cursor < end_ptr && *cursor != b'"' {
                            cursor = cursor.add(if *cursor == b'\\' { 2 } else { 1 });
                        }
                        let end_right = cursor;
                        cursor = cursor.add(1);

                        let left_raw = std::slice::from_raw_parts(start_left, (end_left as usize) - (start_left as usize));
                        let right_raw = std::slice::from_raw_parts(start_right, (end_right as usize) - (start_right as usize));

                        let mut clean_left = [0u8; 128];
                        let mut clen_l = 0;
                        let mut si = 0;
                        while si < left_raw.len() {
                            if left_raw[si] == b'\\' && si + 1 < left_raw.len() {
                                clean_left[clen_l] = match left_raw[si + 1] {
                                    b'"' => b'"', b'\\' => b'\\', b'n' => b'\n', b't' => b'\t', b'r' => b'\r',
                                    other => other,
                                };
                                si += 2;
                            } else {
                                clean_left[clen_l] = left_raw[si];
                                si += 1;
                            }
                            clen_l += 1;
                        }
                        let left_hash = FactoryUtils::fxhash64(&clean_left[..clen_l]);
                        let id1 = text_to_id_table.find(left_hash);

                        let mut clean_right = [0u8; 128];
                        let mut clen_r = 0;
                        let mut si = 0;
                        while si < right_raw.len() {
                            if right_raw[si] == b'\\' && si + 1 < right_raw.len() {
                                clean_right[clen_r] = match right_raw[si + 1] {
                                    b'"' => b'"', b'\\' => b'\\', b'n' => b'\n', b't' => b'\t', b'r' => b'\r',
                                    other => other,
                                };
                                si += 2;
                            } else {
                                clean_right[clen_r] = right_raw[si];
                                si += 1;
                            }
                            clen_r += 1;
                        }
                        let right_hash = FactoryUtils::fxhash64(&clean_right[..clen_r]);
                        let id2 = text_to_id_table.find(right_hash);

                        // ПЕРЕХВАТ ПРИ СБОРКЕ: Ловим пару токенов 140 и 119
                        if (id1 == 140 && id2 == 119) || (clean_left[..clen_l] == *b"\xC3\x90" && clean_right[..clen_r] == *b"\xBB") {
                            println!("[ОРАКУЛ-ФАБРИКА] Мёрдж обнаружен в JSON!");
                            println!("  |-> Левый:  строка='{}', id={:?}", String::from_utf8_lossy(&clean_left[..clen_l]), id1);
                            println!("  |-> Правый: строка='{}', id={:?}", String::from_utf8_lossy(&clean_right[..clen_r]), id2);
                        }

                        if id1 != u32::MAX && id2 != u32::MAX {
                            let mut full_buf = [0u8; 256];
                            std::ptr::copy_nonoverlapping(clean_left.as_ptr(), full_buf.as_mut_ptr(), clen_l);
                            std::ptr::copy_nonoverlapping(clean_right.as_ptr(), full_buf.as_mut_ptr().add(clen_l), clen_r);

                            let full_len = clen_l + clen_r;
                            let full_hash = FactoryUtils::fxhash64(&full_buf[..full_len]);
                            let mid = text_to_id_table.find(full_hash);

                            if id1 == 140 && id2 == 119 {
                                println!("  |-> Результат слияния (mid): id={:?}", mid);
                            }

                            if mid != u32::MAX {
                                let pack = ((id1 as u64) << 32) | (id2 as u64);
                                raw_pairs.push((pack, (rank, mid)));
                                rank += 1;
                            }
                        }

                        while cursor < end_ptr && *cursor != b']' { cursor = cursor.add(1); }
                        if cursor < end_ptr { cursor = cursor.add(1); }
                    }
                }
                println!("[ОРАКУЛ-ФАБРИКА] Всего распаршено пар в raw_pairs: {}", raw_pairs.len());
            }

            // =================================================================
            // --- ЭТАП 3: EOS ID SCAN ЧЕРЕЗ АППАРАТНЫЙ SIMD MATCH ---
            // =================================================================
            cursor = base; // Жесткий гарантированный сброс указателя
            let mut eos_id = 248044;
            let mut added_found = false;

            let target_16 = *b"\"added_tokens\":\0";
            let target_vec = std::arch::x86_64::_mm_loadu_si128(target_16.as_ptr() as *const std::arch::x86_64::__m128i);

            while cursor.add(16) < end_ptr {
                let file_vec = std::arch::x86_64::_mm_loadu_si128(cursor as *const std::arch::x86_64::__m128i);
                let cmp = std::arch::x86_64::_mm_cmpeq_epi8(file_vec, target_vec);
                let mask = std::arch::x86_64::_mm_movemask_epi8(cmp) as u32;

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
                let eos_vec = std::arch::x86_64::_mm_loadu_si128(eos_16.as_ptr() as *const std::arch::x86_64::__m128i);

                while off_ptr.add(16) < end_ptr {
                    let file_vec = std::arch::x86_64::_mm_loadu_si128(off_ptr as *const std::arch::x86_64::__m128i);
                    let cmp = std::arch::x86_64::_mm_cmpeq_epi8(file_vec, eos_vec);
                    let mask = std::arch::x86_64::_mm_movemask_epi8(cmp) as u32;

                    if (mask & 0x1FFF) == 0x1FFF {
                        let mut s = off_ptr;
                        let scan_limit = s.add(200);
                        while s < end_ptr && s < scan_limit {
                            if std::ptr::read(s as *const u32) == u32::from_le_bytes(*b"\"id\"") {
                                s = s.add(4);
                                while s < end_ptr && !(*s).is_ascii_digit() { s = s.add(1); }
                                let start_d = s;
                                while s < end_ptr && (*s).is_ascii_digit() { s = s.add(1); }

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

            // Возвращаем полностью готовый токенизатор, обёрнутый в Box
            Ok(Box::new(BpeTokenizer::new(&raw_pairs, byte_fallback, eos_id, vocab_size, &vocab_compiled_tokens)))
        }
    }
}
