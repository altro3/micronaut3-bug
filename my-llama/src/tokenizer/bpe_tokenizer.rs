pub struct BpeTokenizer {
    pub(crate) keys_flat: Vec<u64>,
    pub(crate) values_flat: Vec<u64>,
    pub(crate) hash_mask: u64,
    pub(crate) byte_pair_ranks: [u64; 65536],
    pub(crate) byte_fallback: [u32; 256],
    pub(crate) id_to_byte: [u8; 512],
    pub eos_token_id: u32,
    pub(crate) vocab_size: usize,
}

impl BpeTokenizer {
    pub fn new(raw_pairs: &[(u64, (u32, u32))], byte_fallback: [u32; 256], eos_token_id: u32, vocab_size: usize) -> Self {
        let mut byte_pair_ranks = [u64::MAX; 65536];
        let mut id_to_byte = [0xFFu8; 512];

        let required_size = raw_pairs.len() * 2;
        let table_size = required_size.max(65536).next_power_of_two();
        let hash_mask = (table_size - 1) as u64;

        let mut keys_flat = vec![u64::MAX; table_size];
        let mut values_flat = vec![u64::MAX; table_size];

        for b in 0..=255 {
            let id = byte_fallback[b] as usize;
            if id < 512 {
                id_to_byte[id] = b as u8;
            }
        }

        for &(pack, (rank, id)) in raw_pairs.iter() {
            let packed_val = ((rank as u64) << 32) | (id as u64);

            let mut h = pack.wrapping_mul(0x517cc1b727220a95);
            h ^= h >> 47;

            let mut target_idx = (h & hash_mask) as usize;
            while keys_flat[target_idx] != u64::MAX && keys_flat[target_idx] != pack {
                target_idx = (target_idx + 1) & (table_size - 1);
            }
            keys_flat[target_idx] = pack;
            values_flat[target_idx] = packed_val;

            let left = (pack >> 32) as u32;
            let right = pack as u32;

            if left < 512 && right < 512 {
                let b1 = id_to_byte[left as usize];
                let b2 = id_to_byte[right as usize];
                if b1 != 0xFF && b2 != 0xFF {
                    byte_pair_ranks[((b1 as usize) << 8) | (b2 as usize)] = packed_val;
                }
            }
        }

        Self {
            keys_flat,
            values_flat,
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
        let is_low = ((left | right) < 512) as u64;

        let low_idx_left = (left as usize) & 511;
        let low_idx_right = (right as usize) & 511;

        let b1 = unsafe { *self.id_to_byte.get_unchecked(low_idx_left) };
        let b2 = unsafe { *self.id_to_byte.get_unchecked(low_idx_right) };

        let is_valid_bytes = ((b1 | b2) != 0xFF) as u64;

        let fast_path_mask = is_low & is_valid_bytes;

        if fast_path_mask != 0 {
            let flat_idx = ((b1 as usize) << 8) | (b2 as usize);
            return unsafe { *self.byte_pair_ranks.get_unchecked(flat_idx) };
        }

        let pack = ((left as u64) << 32) | (right as u64);
        let mut h = pack.wrapping_mul(0x517cc1b727220a95);
        h ^= h >> 47;

        let mask = self.hash_mask as usize;
        let mut idx = (h as usize) & mask;

        unsafe {
            let k0 = *self.keys_flat.get_unchecked(idx);
            if k0 == pack {
                return *self.values_flat.get_unchecked(idx);
            }
            if k0 == u64::MAX {
                return u64::MAX;
            }
            idx = (idx + 1) & mask;

            let k1 = *self.keys_flat.get_unchecked(idx);
            if k1 == pack {
                return *self.values_flat.get_unchecked(idx);
            }
            if k1 == u64::MAX {
                return u64::MAX;
            }
            idx = (idx + 1) & mask;

            let k2 = *self.keys_flat.get_unchecked(idx);
            if k2 == pack {
                return *self.values_flat.get_unchecked(idx);
            }
            if k2 == u64::MAX {
                return u64::MAX;
            }
            idx = (idx + 1) & mask;
        }

        loop {
            let key = unsafe { *self.keys_flat.get_unchecked(idx) };
            if key == pack {
                return unsafe { *self.values_flat.get_unchecked(idx) };
            }
            if key == u64::MAX {
                return u64::MAX;
            }
            idx = (idx + 1) & mask;
        }
    }
}

unsafe impl Send for BpeTokenizer {}
unsafe impl Sync for BpeTokenizer {}
