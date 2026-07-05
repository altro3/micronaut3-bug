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
    ) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
        let weight = word.weight;
        let mut deleted_pairs = Vec::new();
        let mut added_pairs = Vec::new();

        let mut has_pair = false;
        for window in word.tokens.windows(2) {
            if window[0] == pair.0 && window[1] == pair.1 {
                has_pair = true;
                break;
            }
        }
        if !has_pair {
            return (deleted_pairs, added_pairs);
        }

        for window in word.tokens.windows(2) {
            let p = (window[0], window[1]);
            deleted_pairs.push(p);
            if let Some(cnt) = index.pair_counts.get_mut(&p) {
                *cnt -= weight;
            }
        }

        let mut w = 0;
        let mut r = 0;
        let len = word.tokens.len();
        while r < len {
            if r < len - 1 && word.tokens[r] == pair.0 && word.tokens[r + 1] == pair.1 {
                word.tokens[w] = new_id;
                w += 1;
                r += 2;
            } else {
                word.tokens[w] = word.tokens[r];
                w += 1;
                r += 1;
            }
        }
        word.tokens.truncate(w);

        for window in word.tokens.windows(2) {
            let p = (window[0], window[1]);
            added_pairs.push(p);
            *index.pair_counts.entry(p).or_insert(0) += weight;
        }

        if word_idx % 200000 == 0 {
            println!(
                "        [ВОРКЕР ТРАССИРОВКА] Слово Id: {:<7} | Стало токенов: {}",
                word_idx,
                word.tokens.len()
            );
        }

        (deleted_pairs, added_pairs)
    }
}
