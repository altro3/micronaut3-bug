use std::hash::{BuildHasher, Hasher};

#[derive(Default, Clone, Copy)]
pub struct TrainerIdentityHasher {
    hash: u64,
}

impl Hasher for TrainerIdentityHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.hash
    }
    #[inline(always)]
    fn write(&mut self, _b: &[u8]) {}
    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.hash = i.wrapping_mul(0x517cc1b727220a95);
    }
}

#[derive(Default, Clone, Copy)]
pub struct BuildTrainerHasher;

impl BuildHasher for BuildTrainerHasher {
    type Hasher = TrainerIdentityHasher;
    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        TrainerIdentityHasher { hash: 0 }
    }
}

pub struct BpeWorker {
    pub words: Vec<Vec<u32>>,
    pub word_counts: Vec<u32>,
    pub table_keys: Vec<u64>,
    pub table_stats: Vec<i64>,
    pub table_heads: Vec<u32>,
    pub next_node: Vec<u32>,
    pub mask: usize,
}

impl BpeWorker {
    pub fn with_capacity(words: Vec<Vec<u32>>, word_counts: Vec<u32>, table_size: usize) -> Self {
        let mask = table_size - 1;
        let max_words = words.len();
        let mut worker = Self {
            words,
            word_counts,
            table_keys: vec![u64::MAX; table_size],
            table_stats: vec![0; table_size],
            table_heads: vec![u32::MAX; table_size],
            next_node: vec![u32::MAX; max_words * 8],
            mask,
        };

        let words_base_ptr = worker.words.as_ptr();
        let counts_base_ptr = worker.word_counts.as_ptr();

        for w_idx in 0..max_words {
            unsafe {
                let word_ptr = words_base_ptr.add(w_idx);
                let len = (*word_ptr).len();
                if len < 2 {
                    continue;
                }

                let weight = *counts_base_ptr.add(w_idx) as i64;
                let data_ptr = (*word_ptr).as_ptr();

                for i in 0..len - 1 {
                    let pack = ((*data_ptr.add(i)) as u64) << 32 | (*data_ptr.add(i + 1)) as u64;
                    worker.insert_initial(pack, weight, w_idx as u32);
                }
            }
        }
        worker
    }

    #[inline(always)]
    fn insert_initial(&mut self, pack: u64, weight: i64, w_idx: u32) {
        let mut idx = (pack.wrapping_mul(0x517cc1b727220a95) as usize) & self.mask;
        loop {
            if self.table_keys[idx] == pack {
                self.table_stats[idx] += weight;
                return;
            }
            if self.table_keys[idx] == u64::MAX {
                self.table_keys[idx] = pack;
                self.table_stats[idx] = weight;
                let old_head = self.table_heads[idx];
                self.table_heads[idx] = w_idx;
                self.next_node[w_idx as usize] = old_head;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    pub unsafe fn merge_tokens_inplace(&mut self, w_idx: usize, id1: u32, id2: u32, new_id: u32, weight: i64) {
        let word = self.words.get_unchecked_mut(w_idx);
        let len = word.len();
        if len < 2 {
            return;
        }
        let (mut r_ptr, mut w_ptr, end_ptr, mut new_len) = (word.as_ptr(), word.as_mut_ptr(), word.as_ptr().add(len), 0);

        while r_ptr < end_ptr {
            if r_ptr.add(1) < end_ptr && *r_ptr == id1 && *r_ptr.add(1) == id2 {
                *w_ptr = new_id;
                r_ptr = r_ptr.add(2);
            } else {
                *w_ptr = *r_ptr;
                r_ptr = r_ptr.add(1);
            }
            w_ptr = w_ptr.add(1);
            new_len += 1;
        }
        word.set_len(new_len);

        if new_len >= 2 {
            for i in 0..new_len - 1 {
                let curr_id = *word.get_unchecked(i);
                let next_id = *word.get_unchecked(i + 1);
                if curr_id == new_id || next_id == new_id {
                    let pack = ((curr_id as u64) << 32) | (next_id as u64);
                    let mut idx = (pack.wrapping_mul(0x517cc1b727220a95) as usize) & self.mask;
                    loop {
                        if self.table_keys[idx] == pack {
                            self.table_stats[idx] += weight;
                            break;
                        }
                        if self.table_keys[idx] == u64::MAX {
                            self.table_keys[idx] = pack;
                            self.table_stats[idx] = weight;
                            let old_head = self.table_heads[idx];
                            self.table_heads[idx] = w_idx as u32;
                            self.next_node[w_idx] = old_head;
                            break;
                        }
                        idx = (idx + 1) & self.mask;
                    }
                }
            }
        }
    }
}
