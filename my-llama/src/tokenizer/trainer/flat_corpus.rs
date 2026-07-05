use std::collections::{HashMap, HashSet};

pub struct WordEntry {
    pub tokens: Vec<u32>,
    pub weight: i64,
}

pub struct FlatCorpus {
    pub words: Vec<WordEntry>,
}

pub struct PairSpanIndex {
    pub pair_counts: HashMap<(u32, u32), i64>,
    pub pair_index: HashMap<(u32, u32), HashSet<usize>>,
}

impl FlatCorpus {
    pub fn build(unique_words: HashMap<Vec<u8>, usize>) -> (Self, PairSpanIndex) {
        let mut words = Vec::with_capacity(unique_words.len());
        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::with_capacity(unique_words.len() * 2);
        let mut pair_index: HashMap<(u32, u32), HashSet<usize>> = HashMap::with_capacity(unique_words.len() * 2);

        for (bytes, count) in unique_words {
            if bytes.is_empty() {
                continue;
            }
            let weight = count as i64;

            // Переводим в строку БЕЗОПАСНО для проверки Unicode-метрик
            let word_str = String::from_utf8_lossy(&bytes);

            // ЗАЩИТА 1: Если регулярка выдала кусок длиннее 16 реальных символов,
            // мы принудительно нарезаем его по границам char, изолируя их.
            if word_str.chars().count() > 16 {
                for c in word_str.chars() {
                    let mut char_buf = [0u8; 4];
                    let char_bytes = c.encode_utf8(&mut char_buf).as_bytes();

                    // Пушим в корпус строго байты (0..255), сохраняя инвариант BPE!
                    words.push(WordEntry {
                        tokens: char_bytes.iter().map(|&b| b as u32).collect(),
                        weight,
                    });
                }
                continue;
            }

            // Для нормальных чистых слов переводим байты в u32-токены базового пространства (0..255)
            let word_tokens: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
            words.push(WordEntry { tokens: word_tokens, weight });
        }

        // Вспомогательная мапа: быстрое восстановление байт для базовых токенов 0..255
        let base_id_to_byte: Vec<u8> = (0..256).map(|b| b as u8).collect();

        // Строим начальный индекс и частоты пар
        for (w_idx, word) in words.iter().enumerate() {
            if word.tokens.len() < 2 {
                continue;
            }

            for window in word.tokens.windows(2) {
                let pair = (window[0], window[1]);

                // ЗАЩИТА 2: Фильтрация склейки знаков препинания на старте (для базовых токенов)
                if pair.0 < 256 && pair.1 < 256 {
                    let b1 = base_id_to_byte[pair.0 as usize];
                    let b2 = base_id_to_byte[pair.1 as usize];

                    // Пытаемся интерпретировать байты как ASCII символы для проверки пунктуации
                    let c1 = b1 as char;
                    let c2 = b2 as char;

                    // Если один из базовых токенов является ASCII-знаком препинания (точка, запятая, дефис и т.д.)
                    if c1.is_ascii_punctuation() || c2.is_ascii_punctuation() {
                        // Исключение: разрешаем пробелы и ведущий пробел Qwen (0x20)
                        if b1 != 0x20 && b1 != 0x0A && b1 != 0x0D && b1 != 0x09 {
                            continue;
                        }
                    }
                }

                *pair_counts.entry(pair).or_insert(0) += word.weight;
                pair_index.entry(pair).or_insert_with(HashSet::new).insert(w_idx);
            }
        }

        (Self { words }, PairSpanIndex { pair_counts, pair_index })
    }
}
