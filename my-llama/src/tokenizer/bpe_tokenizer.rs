pub struct BpeTokenizer {
    pub(crate) pair_ranks_flat: Vec<u64>,
    pub(crate) hash_mask: u64,
    pub(crate) byte_pair_ranks: [u64; 65536],
    pub(crate) byte_fallback: [u32; 256],
    pub(crate) id_to_byte: [i16; 513],
    pub eos_token_id: u32,
    pub(crate) vocab_size: usize,
}

impl BpeTokenizer {
    pub fn new(raw_pair_ranks: rustc_hash::FxHashMap<u64, u32>, byte_fallback: [u32; 256], eos_token_id: u32, vocab_size: usize) -> Self {
        let mut byte_pair_ranks = [u64::MAX; 65536];
        let mut id_to_byte = [-1i16; 513];

        let required_size = raw_pair_ranks.len() * 4;

        let table_size = required_size.max(65536).next_power_of_two();
        let hash_mask = (table_size - 1) as u64;

        let mut pair_ranks_flat = vec![u64::MAX; table_size];

        for b in 0..=255 {
            let id = byte_fallback[b] as usize;
            if id < 512 && id < vocab_size {
                id_to_byte[id] = b as i16;
            }
        }

        for (&pack, &rank) in raw_pair_ranks.iter() {
            let left = (pack >> 32) as u32;
            let right = pack as u32;

            let token_id = 0u32;
            let packed_val = ((rank as u64) << 32) | (token_id as u64);

            let mut h = pack ^ (pack >> 33);
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h = h ^ (h >> 33);

            let idx = (h & hash_mask) as usize;

            let mut target_idx = idx;
            while pair_ranks_flat[target_idx] != u64::MAX {
                target_idx = (target_idx + 1) & (table_size - 1);
            }
            pair_ranks_flat[target_idx] = packed_val;

            let b1 = if left < 512 { id_to_byte[left as usize] } else { -1 };
            let b2 = if right < 512 { id_to_byte[right as usize] } else { -1 };

            if b1 >= 0 && b2 >= 0 {
                byte_pair_ranks[((b1 as usize) << 8) | (b2 as usize)] = packed_val;
            }
        }

        Self {
            pair_ranks_flat,
            hash_mask,
            byte_pair_ranks,
            byte_fallback,
            id_to_byte,
            eos_token_id,
            vocab_size,
        }
    }

    #[inline(always)]
    pub(crate) fn get_pair_packed(&self, left: u32, right: u32) -> u64 {
        let idx_left = if left < 512 { left as usize } else { 512 };
        let idx_right = if right < 512 { right as usize } else { 512 };

        let b1 = unsafe { *self.id_to_byte.get_unchecked(idx_left) };
        let b2 = unsafe { *self.id_to_byte.get_unchecked(idx_right) };

        if (b1 >= 0) & (b2 >= 0) {
            let flat_idx = ((b1 as usize) << 8) | (b2 as usize);
            return unsafe { *self.byte_pair_ranks.get_unchecked(flat_idx) };
        }

        let pack = ((left as u64) << 32) | (right as u64);
        let mut h = pack ^ (pack >> 33);
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h = h ^ (h >> 33);
        let idx = (h & self.hash_mask) as usize;

        unsafe { *self.pair_ranks_flat.get_unchecked(idx) }
    }
}
