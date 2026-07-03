use crate::tokenizer::BpeTokenizer;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

#[derive(Deserialize)]
struct BpeModelFields {
    vocab: HashMap<String, u32>,
    merges: Vec<[String; 2]>,
}

#[derive(Deserialize)]
struct QwenJsonModel {
    added_tokens: Option<Vec<AddedToken>>,
    model: BpeModelFields,
}

pub struct TokenizerFactory;

impl TokenizerFactory {
    /// Безопасная, пуленепробиваемая фабрика на Serde, полностью адаптированная под твой дамп JSON.
    pub fn from_file(file_path: &str) -> std::io::Result<Box<BpeTokenizer>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        println!("[ФАБРИКА-SERDE] Начинаю высокоскоростной разбор JSON через Serde...");
        let root_model: QwenJsonModel = serde_json::from_reader(reader).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Извлекаем автоматический eos_token_id, если он есть в added_tokens
        let mut eos_id = 248044; // Дефолтный фоллбэк
        if let Some(tokens) = root_model.added_tokens {
            for tok in tokens {
                if tok.content == "<|endoftext|>" {
                    eos_id = tok.id;
                    break;
                }
            }
        }

        let json_model = &root_model.model;
        let vocab_size = json_model.vocab.len();
        let mut byte_fallback = [0u32; 256];
        let mut raw_pairs = Vec::with_capacity(json_model.merges.len());

        let mut vocab_compiled_tokens = vec![Vec::new(); vocab_size.max(260000)];

        println!("[ФАБРИКА-SERDE] Компилирую вокабуляр и byte_fallback...");
        for (token_str, &id) in json_model.vocab.iter() {
            // Переводим Unicode строку Serde в сырые Latin-1 байты
            let clean_bytes = Self::string_to_latin1_bytes(token_str);
            let id_idx = id as usize;

            if id_idx >= vocab_compiled_tokens.len() {
                vocab_compiled_tokens.resize(id_idx + 1, Vec::new());
            }
            vocab_compiled_tokens[id_idx] = clean_bytes.clone();

            // ИСПРАВЛЕНО: Если токен после конвертации в Latin-1 весит ровно 1 байт —
            // это и есть истинный одиночный байт-фоллбэк BPE-модели!
            if clean_bytes.len() == 1 {
                let byte_idx = clean_bytes[0] as usize;
                // Записываем ID, только если этот слот еще не был занят более приоритетным токеном
                if byte_fallback[byte_idx] == 0 {
                    byte_fallback[byte_idx] = id;
                }
            }
        }

        // Логика заполнения дефолтов для непечатных ASCII, которых может не быть в vocab
        for b in 0..=255 {
            if byte_fallback[b] == 0 {
                byte_fallback[b] = b as u32;
            }
        }
        // 2. СБОРКА ПРАВИЛ МЁРДЖЕЙ (Парсим merges из двумерного массива пар строк)
        println!("[ФАБРИКА-SERDE] Компилирую графы слияний (всего пар: {})...", json_model.merges.len());
        for (rank, pair) in json_model.merges.iter().enumerate() {
            let left_str = &pair[0];
            let right_str = &pair[1];

            if let (Some(&id1), Some(&id2)) = (json_model.vocab.get(left_str), json_model.vocab.get(right_str)) {
                // Склеиваем строки на уровне валидного Rust Unicode
                let mut full_str = String::with_capacity(left_str.len() + right_str.len());
                full_str.push_str(left_str);
                full_str.push_str(right_str);

                // Проверяем, существует ли такой результирующий токен в словаре
                if let Some(&mid) = json_model.vocab.get(&full_str) {
                    let pack = ((id1 as u64) << 32) | (id2 as u64);
                    raw_pairs.push((pack, (rank as u32, mid)));
                }
            }
        }

        println!("[ФАБРИКА-SERDE] Успех! Всего упаковано валидных правил в raw_pairs: {}", raw_pairs.len());

        // Возвращаем полностью готовый, обёрнутый в Box токенизатор
        Ok(Box::new(BpeTokenizer::new(
            &raw_pairs,
            byte_fallback,
            eos_id,
            vocab_size,
            &vocab_compiled_tokens,
        )))
    }

    /// Вспомогательный метод: переводит Unicode-строку Serde обратно в сырые Latin-1 байты.
    /// Гарантирует, что обратный вокабуляр внутри BpeTokenizer будет лежать байт-в-байт как в оригинале.
    fn string_to_latin1_bytes(s: &str) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(s.len());
        for c in s.chars() {
            bytes.push(c as u32 as u8);
        }
        bytes
    }

    /// Оригинальный HuggingFace / Qwen демаппинг Latin-1 сдвигов в сырые байты текста.
    #[inline(always)]
    fn decode_inplace_hf(s: &[u8], buf: &mut [u8]) -> usize {
        if s.is_empty() {
            return 0;
        }
        let b0 = s[0];
        let mut cp: u32 = 0;
        let len = s.len();

        if b0 < 0x80 {
            cp = b0 as u32;
        } else if (b0 & 0xE0) == 0xC0 && len >= 2 {
            cp = (((b0 & 0x1F) as u32) << 6) | (s[1] & 0x3F) as u32;
        } else if (b0 & 0xF0) == 0xE0 && len >= 3 {
            cp = (((b0 & 0x0F) as u32) << 12) | (((s[1] & 0x3F) as u32) << 6) | (s[2] & 0x3F) as u32;
        } else if (b0 & 0xF8) == 0xF0 && len >= 4 {
            cp = (((b0 & 0x07) as u32) << 18) | (((s[1] & 0x3F) as u32) << 12) | (((s[2] & 0x3F) as u32) << 6) | (s[3] & 0x3F) as u32;
        }

        let raw_byte = match cp {
            0x00..=0x7F => cp as u8,
            0x00A0..=0x00FF => cp as u8,
            0x0100..=0x011F => (cp - 0x0100) as u8,
            0x0120..=0x013F => (cp - 0x0120 + 127) as u8,
            0x0140 => 173,
            _ => cp as u8,
        };

        buf[0] = raw_byte;
        1
    }
}
