pub const HASH_SIZE: usize = 524288; // 2^19 — степень двойки для мгновенного битового маскирования
pub const HASH_MASK: usize = HASH_SIZE - 1;

pub struct InlineHashTable {
    pub keys: Vec<u64>,
    pub vals: Vec<u32>,
}

impl InlineHashTable {
    pub fn new() -> Self {
        Self {
            keys: vec![u64::MAX; HASH_SIZE],
            vals: vec![u32::MAX; HASH_SIZE],
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, hash: u64, id: u32) {
        let mut idx = (hash as usize) & HASH_MASK;
        while self.keys[idx] != u64::MAX {
            idx = (idx + 1) & HASH_MASK;
        }
        self.keys[idx] = hash;
        self.vals[idx] = id;
    }

    #[inline(always)]
    pub fn find(&self, hash: u64) -> u32 {
        let mut idx = (hash as usize) & HASH_MASK;
        loop {
            if self.keys[idx] == hash {
                return self.vals[idx];
            }
            if self.keys[idx] == u64::MAX {
                return u32::MAX;
            }
            idx = (idx + 1) & HASH_MASK;
        }
    }
}
