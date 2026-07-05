use crate::tokenizer::trainer::flat_corpus::WordEntry;
use std::collections::HashMap;

pub struct ThreadDeltaWorker;

impl ThreadDeltaWorker {
    #[inline(always)]
    pub fn process_word_locally(
        word: &mut WordEntry,
        targets: &[((u32, u32), u32)], // Корректно упорядоченный по частоте ТОП батча
        local_delta: &mut HashMap<(u32, u32), i64>,
    ) {
        if word.tokens.len() < 2 { return; }

        let weight = word.weight;
        let mut mutated = false;

        let mut new_tokens = Vec::with_capacity(word.tokens.len());
        let mut i = 0;

        while i < word.tokens.len() {
            if i < word.tokens.len() - 1 {
                let current_pair = (word.tokens[i], word.tokens[i + 1]);

                // Ищем, матчится ли текущее окно. Поиск идет линейно по targets.
                // Так как targets содержит всего 256 элементов и лежит в L1-кэше ядра,
                // этот поиск выполняется процессором Intel Ultra 9 мгновенно!
                if let Some(&(_, new_id)) = targets.iter().find(|&&(pair, _)| pair == current_pair) {

                    // 1. Старая пара уничтожается
                    *local_delta.entry(current_pair).or_insert(0) -= weight;

                    // 2. Убираем и пересчитываем левый контекст
                    if !new_tokens.is_empty() {
                        let left_token = new_tokens[new_tokens.len() - 1];
                        *local_delta.entry((left_token, current_pair.0)).or_insert(0) -= weight;
                        *local_delta.entry((left_token, new_id)).or_insert(0) += weight;
                    }

                    // 3. Убираем и пересчитываем правый контекст
                    if i + 2 < word.tokens.len() {
                        let right_token = word.tokens[i + 2];
                        *local_delta.entry((current_pair.1, right_token)).or_insert(0) -= weight;
                        *local_delta.entry((new_id, right_token)).or_insert(0) += weight;
                    }

                    new_tokens.push(new_id);
                    i += 2; // Схлопнули 2 токена, прыгаем дальше
                    mutated = true;
                    continue;
                }
            }
            new_tokens.push(word.tokens[i]);
            i += 1;
        }

        if mutated {
            word.tokens = new_tokens;
        }
    }
}
