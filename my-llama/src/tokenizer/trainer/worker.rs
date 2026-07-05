use crate::tokenizer::trainer::flat_corpus::WordEntry;
use std::collections::{HashMap, HashSet};

pub struct ThreadDeltaWorker;

impl ThreadDeltaWorker {
    /// Локальный мёрж внутри ОДНОГО слова с фиксацией изменений во временный буфер дельт
    #[inline(always)]
    pub fn merge_pair_with_callback<F>(
        word: &mut WordEntry,
        target_pair: (u32, u32),
        new_id: u32,
        mut on_pair_change: F,
    ) where
        F: FnMut((u32, u32), i64),
    {
        if word.tokens.len() < 2 { return; }

        let weight = word.weight;
        let mut i = 0;
        let mut mutated = false;
        let mut new_tokens = Vec::with_capacity(word.tokens.len());

        while i < word.tokens.len() {
            if i < word.tokens.len() - 1 && (word.tokens[i], word.tokens[i + 1]) == target_pair {
                // Старая пара уничтожена
                on_pair_change(target_pair, -weight);

                // Корректируем левый контекст
                if !new_tokens.is_empty() {
                    let left_token = new_tokens[new_tokens.len() - 1];
                    on_pair_change((left_token, target_pair.0), -weight);
                    on_pair_change((left_token, new_id), weight);
                }

                // Корректируем правый контекст
                if i + 2 < word.tokens.len() {
                    let right_token = word.tokens[i + 2];
                    on_pair_change((target_pair.1, right_token), -weight);
                    on_pair_change((new_id, right_token), weight);
                }

                new_tokens.push(new_id);
                i += 2;
                mutated = true;
                continue;
            }
            new_tokens.push(word.tokens[i]);
            i += 1;
        }

        if mutated {
            word.tokens = new_tokens;
        }
    }
}
