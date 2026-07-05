use std::collections::HashMap;

pub struct FlatCorpus {
    pub tokens: Vec<u32>,
    pub next: Vec<i32>,
    pub prev: Vec<i32>,
    pub word_starts: Vec<usize>,
    pub word_counts: Vec<usize>,
}

impl FlatCorpus {
    pub fn build(unique_words: HashMap<Vec<u8>, usize>) -> (Self, HashMap<(u32, u32), i64>) {
        let total_tokens: usize = unique_words.keys().map(|w| w.len()).sum();
        let total_words = unique_words.len();

        let mut corpus = Self {
            tokens: Vec::with_capacity(total_tokens),
            next: Vec::with_capacity(total_tokens),
            prev: Vec::with_capacity(total_tokens),
            word_starts: Vec::with_capacity(total_words),
            word_counts: Vec::with_capacity(total_words),
        };

        let mut pair_stats: HashMap<(u32, u32), i64> = HashMap::with_capacity(total_words * 2);
        let mut current_offset = 0;

        for (bytes, count) in unique_words {
            let w_len = bytes.len();
            corpus.word_starts.push(current_offset);
            corpus.word_counts.push(count);

            for (i, &b) in bytes.iter().enumerate() {
                corpus.tokens.push(b as u32);
                corpus.next.push(if i == w_len - 1 { -1 } else { (current_offset + i + 1) as i32 });
                corpus.prev.push(if i == 0 { -1 } else { (current_offset + i - 1) as i32 });

                if i < w_len - 1 {
                    let pair = (b as u32, bytes[i + 1] as u32);
                    *pair_stats.entry(pair).or_insert(0) += count as i64;
                }
            }
            current_offset += w_len;
        }

        (corpus, pair_stats)
    }

    #[inline(always)]
    pub fn find_word_index(&self, pos: usize) -> usize {
        self.word_starts.binary_search(&pos).unwrap_or_else(|idx| idx - 1)
    }
}
