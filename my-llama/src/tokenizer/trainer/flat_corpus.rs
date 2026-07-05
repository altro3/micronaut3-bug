use std::collections::HashMap;

pub struct WordEntry {
    pub tokens: Vec<u32>,
    pub weight: i64,
}

pub struct FlatCorpus {
    pub words: Vec<WordEntry>,
}

impl FlatCorpus {
    pub fn build(unique_words: HashMap<Vec<u8>, usize>) -> (Self, HashMap<(u32, u32), i64>) {
        let mut words = Vec::with_capacity(unique_words.len());
        let mut pair_stats: HashMap<(u32, u32), i64> = HashMap::with_capacity(unique_words.len() * 2);

        for (bytes, count) in unique_words {
            if bytes.is_empty() { continue; }
            let weight = count as i64;

            // Если регулярка выдала огромную склейку (больше 24 байт),
            // рассыпаем её на отдельные байты длиною в 1 токен, блокируя мёржи мусора
            if bytes.len() > 24 {
                for &b in &bytes {
                    words.push(WordEntry {
                        tokens: vec![b as u32],
                        weight,
                    });
                }
                continue;
            }

            // Для нормальных, чистых слов собираем токены
            let word_tokens: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

            // ИСПРАВЛЕНИЕ: Явная распаковка слайса windows(2) в кортеж (u32, u32)
            for window in word_tokens.windows(2) {
                let pair = (window[0], window[1]);
                *pair_stats.entry(pair).or_insert(0) += weight;
            }

            words.push(WordEntry {
                tokens: word_tokens,
                weight,
            });
        }

        (Self { words }, pair_stats)
    }
}
