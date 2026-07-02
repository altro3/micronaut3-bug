use super::bpe_types::BpeValue;

pub struct BpeTokenizer {
    pub(crate) keys_flat: Vec<u64>,
    pub(crate) values_flat: Vec<u64>,
    pub(crate) hash_mask: u64,
    pub(crate) byte_pair_ranks: [u64; 65536],
    pub(crate) byte_fallback: [u32; 256],
    pub(crate) id_to_byte: [i16; 512],
    pub eos_token_id: u32,
    pub(crate) vocab_size: usize,
}

impl BpeTokenizer {
    pub fn new(raw_pairs: &[(u64, BpeValue)], byte_fallback: [u32; 256], eos_token_id: u32, vocab_size: usize) -> Self {
        let mut byte_pair_ranks = [u64::MAX; 65536];
        let mut id_to_byte = [-1i16; 512];

        let required_size = raw_pairs.len() * 2;
        let table_size = required_size.max(65536).next_power_of_two();
        let hash_mask = (table_size - 1) as u64;

        let mut keys_flat = vec![u64::MAX; table_size];
        let mut values_flat = vec![u64::MAX; table_size];

        for b in 0..=255 {
            let id = byte_fallback[b] as usize;
            if id < 512 && id < vocab_size {
                id_to_byte[id] = b as i16;
            }
        }

        for &(pack, val) in raw_pairs.iter() {
            let packed_val = ((val.rank as u64) << 32) | (val.id as u64);

            let mut h = pack;
            h ^= h >> 30;
            h = h.wrapping_mul(0xbf58476d1ce4e5b9);
            h ^= h >> 27;

            let idx = (h & hash_mask) as usize;
            let mut target_idx = idx;

            while keys_flat[target_idx] != u64::MAX && keys_flat[target_idx] != pack {
                target_idx = (target_idx + 1) & (table_size - 1);
            }
            keys_flat[target_idx] = pack;
            values_flat[target_idx] = packed_val;

            let left = (pack >> 32) as u32;
            let right = pack as u32;

            let b1 = if left < 512 { id_to_byte[left as usize] } else { -1 };
            let b2 = if right < 512 { id_to_byte[right as usize] } else { -1 };

            if b1 >= 0 && b2 >= 0 {
                byte_pair_ranks[((b1 as usize) << 8) | (b2 as usize)] = packed_val;
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
        let pack = ((left as u64) << 32) | (right as u64);

        let m1 = left < 512;
        let m2 = right < 512;

        let idx_l = (left * m1 as u32) as usize;
        let idx_r = (right * m2 as u32) as usize;

        let b1 = unsafe { *self.id_to_byte.get_unchecked(idx_l) };
        let b2 = unsafe { *self.id_to_byte.get_unchecked(idx_r) };

        let is_valid_byte_pair = (b1 >= 0) & (b2 >= 0) & m1 & m2;

        if is_valid_byte_pair {
            let flat_idx = ((b1 as usize) << 8) | (b2 as usize);
            return unsafe { *self.byte_pair_ranks.get_unchecked(flat_idx) };
        }

        let mut h = pack;
        h ^= h >> 31;
        h ^= h << 21;
        h ^= h >> 4;

        let mask = self.hash_mask as usize;
        let mut idx = (h as usize) & mask;

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
