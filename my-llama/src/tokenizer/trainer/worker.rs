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
            next_node: vec![u32::MAX; max_words],
            mask,
        };

        for w_idx in 0..max_words {
            unsafe {
                let word = worker.words.get_unchecked(w_idx);
                let len = word.len();
                if len < 2 {
                    continue;
                }

                let weight = *worker.word_counts.get_unchecked(w_idx) as i64;
                let data_ptr = word.as_ptr();

                for i in 0..len - 1 {
                    let pack = ((*data_ptr.add(i)) as u64) << 32 | (*data_ptr.add(i + 1)) as u64;
                    worker.insert_initial(pack, weight, w_idx as u32);
                }
            }
        }
        worker
    }

    #[inline(always)]
    pub fn insert_initial(&mut self, pack: u64, weight: i64, w_idx: u32) {
        let mut idx = (pack.wrapping_mul(0x517cc1b727220a95) as usize) & self.mask;
        loop {
            if self.table_keys[idx] == pack {
                self.table_stats[idx] += weight;
                if self.table_heads[idx] != w_idx {
                    let old_head = self.table_heads[idx];
                    self.table_heads[idx] = w_idx;
                    unsafe { *self.next_node.get_unchecked_mut(w_idx as usize) = old_head };
                }
                return;
            }
            if self.table_keys[idx] == u64::MAX {
                self.table_keys[idx] = pack;
                self.table_stats[idx] = weight;
                let old_head = self.table_heads[idx];
                self.table_heads[idx] = w_idx;
                unsafe { *self.next_node.get_unchecked_mut(w_idx as usize) = old_head };
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    pub fn merge_tokens_inplace(&mut self, w_idx: usize, id1: u32, id2: u32, new_id: u32, weight: i64) {
        let word = unsafe { self.words.get_unchecked_mut(w_idx) };
        let len = word.len();
        if len < 2 { return; }

        let table_len = self.table_keys.len();

        // --- ШАГ 1: ТОЧЕЧНОЕ ВЫЧИТАНИЕ С ЗАЩИТОЙ ---
        for i in 0..len - 1 {
            unsafe {
                let id_curr = *word.get_unchecked(i) as u64;
                let id_next = *word.get_unchecked(i + 1) as u64;
                let pack = (id_curr << 32) | id_next;

                let mut idx = (pack.wrapping_mul(0x517cc1b727220a95) as usize) & self.mask;
                let mut steps = 0;
                loop {
                    if self.table_keys[idx] == pack {
                        self.table_stats[idx] -= weight;
                        break;
                    }
                    if self.table_keys[idx] == u64::MAX { break; }

                    steps += 1;
                    if steps >= table_len {
                        panic!("[ПАНИКА BPE] Вечный цикл на Шаге 1! Таблица воркера переполнена при вычитании пары.");
                    }
                    idx = (idx + 1) & self.mask;
                }
            }
        }

        // --- ШАГ 2: PTR-СЛИЯНИЕ (БЕЗ ИЗМЕНЕНИЙ) ---
        let mut r_ptr = word.as_ptr();
        let mut w_ptr = word.as_mut_ptr();
        let end_ptr = unsafe { word.as_ptr().add(len) };
        let mut new_len = 0;

        unsafe {
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
        }

        // --- ШАГ 3: НАЧИСЛЕНИЕ ВЕСОВ С ГАРАНТИРОВАННОЙ ЗАЩИТОЙ ОТ ЗАЦИКЛИВАНИЯ ---
        if new_len >= 2 {
            for i in 0..new_len - 1 {
                let curr_id = unsafe { *word.get_unchecked(i) };
                let next_id = unsafe { *word.get_unchecked(i + 1) };

                let pack = ((curr_id as u64) << 32) | (next_id as u64);
                let mut idx = (pack.wrapping_mul(0x517cc1b727220a95) as usize) & self.mask;
                let mut steps = 0;

                loop {
                    if self.table_keys[idx] == pack {
                        self.table_stats[idx] += weight;

                        // ИСПРАВЛЕНИЕ: Добавляем слово в цепочку heads ТОЛЬКО если
                        // его там еще нет на текущей итерации.
                        // Защищает от ситуации, когда пара дублируется в одном слове
                        // и замыкает next_node[w_idx] = w_idx.
                        if self.table_heads[idx] != w_idx as u32 {
                            let old_head = self.table_heads[idx];
                            self.table_heads[idx] = w_idx as u32;
                            unsafe { *self.next_node.get_unchecked_mut(w_idx) = old_head };
                        }
                        break;
                    }
                    if self.table_keys[idx] == u64::MAX {
                        self.table_keys[idx] = pack;
                        self.table_stats[idx] = weight;

                        // ИСПРАВЛЕНИЕ аналогично: защита при создании новой ячейки
                        if self.table_heads[idx] != w_idx as u32 {
                            let old_head = self.table_heads[idx];
                            self.table_heads[idx] = w_idx as u32;
                            unsafe { *self.next_node.get_unchecked_mut(w_idx) = old_head };
                        }
                        break;
                    }

                    steps += 1;
                    if steps >= table_len {
                        panic!("[ПАНИКА BPE] Переполнение хэш-таблицы на Шаге 3.");
                    }
                    idx = (idx + 1) & self.mask;
                }
            }
        }
    }
}
