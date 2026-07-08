use crate::tokenizer::trainer::position_index::{IsolatedWord, PositionIndex};

pub struct ThreadDeltaWorker;

impl ThreadDeltaWorker {
    #[inline(always)]
    pub fn merge_in_word(
        word: &mut IsolatedWord,
        word_idx: u32,
        pair: (u32, u32),
        new_id: u32,
        index: &mut PositionIndex,
        added_pairs: &mut Vec<(u32, u32)>,
    ) {
        added_pairs.clear();
        let weight = word.weight;
        let tokens = &mut word.tokens;
        let len = tokens.len();

        if len < 2 {
            return;
        }

        let mut has_pair = false;
        for i in 0..len - 1 {
            if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                has_pair = true;
                break;
            }
        }

        if !has_pair {
            return;
        }

        for i in 0..len - 1 {
            let p = (tokens[i], tokens[i + 1]);
            if let Some(cnt) = index.pair_counts.get_mut(&p) {
                *cnt -= weight;
            }
        }

        let mut w = 0;
        let mut r = 0;
        while r < len {
            if r < len - 1 && tokens[r] == pair.0 && tokens[r + 1] == pair.1 {
                tokens[w] = new_id;
                w += 1;
                r += 2;
            } else {
                tokens[w] = tokens[r];
                w += 1;
                r += 1;
            }
        }
        tokens.truncate(w);

        if w >= 2 {
            for i in 0..w - 1 {
                let p = (tokens[i], tokens[i + 1]);
                added_pairs.push(p);
                *index.pair_counts.entry(p).or_insert(0) += weight;
            }
        }

        if word_idx.is_multiple_of(200000) {
            println!("        [ВОРКЕР ТРАССИРОВКА] Слово Id: {:<7} | Стало токенов: {}", word_idx, w);
        }
    }
}
