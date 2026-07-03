use crate::tokenizer::bpe_types::{BpeRank, TokenId};

pub struct BpeTokenizer {
    pub keys_flat: Vec<u64>,
    pub values_flat: Vec<u64>,
    pub hash_mask: u64,
    pub(crate) byte_pair_ranks: [u64; 65536],
    pub byte_fallback: [u32; 256],
    pub(crate) id_to_byte: [u8; 512],
    pub eos_token_id: u32,
    pub(crate) vocab_size: usize,
    pub(crate) vocab_bytes_flat: Vec<u8>,
    pub(crate) vocab_offsets_flat: Vec<u64>,
}

impl BpeTokenizer {
    pub fn new(
        raw_pairs: &[(u64, (BpeRank, TokenId))],
        byte_fallback: [u32; 256],
        eos_token_id: u32,
        vocab_size: usize,
        vocab_compiled_tokens: &[Vec<u8>],
    ) -> Self {
        let mut tmp_ranks = vec![u64::MAX; 65536];
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

            if keys_flat[target_idx] == u64::MAX {
                keys_flat[target_idx] = pack;
                values_flat[target_idx] = packed_val;

                let left = (pack >> 32) as u32;
                let right = pack as u32;

                if left < 512 && right < 512 {
                    let b1 = id_to_byte[left as usize];
                    let b2 = id_to_byte[right as usize];
                    if b1 != 0xFF && b2 != 0xFF {
                        tmp_ranks[((b1 as usize) << 8) | (b2 as usize)] = packed_val;
                    }
                }
            }
        }

        let mut byte_pair_ranks = [u64::MAX; 65536];
        byte_pair_ranks.copy_from_slice(&tmp_ranks);

        let mut vocab_bytes_flat = Vec::with_capacity(vocab_size * 8);
        let mut vocab_offsets_flat = vec![0u64; vocab_compiled_tokens.len().max(vocab_size)];

        for (id, token_bytes) in vocab_compiled_tokens.iter().enumerate() {
            if token_bytes.is_empty() {
                continue;
            }
            let offset = vocab_bytes_flat.len() as u64;
            let length = token_bytes.len() as u64;

            vocab_bytes_flat.extend_from_slice(token_bytes);
            vocab_offsets_flat[id] = (offset << 32) | length;
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
            vocab_bytes_flat,
            vocab_offsets_flat,
        }
    }

    #[inline(always)]
    pub(crate) fn get_pair_packed(&self, left: u32, right: u32) -> u64 {
        if (left | right) < 512 {
            let b1 = unsafe { *self.id_to_byte.get_unchecked(left as usize) };
            let b2 = unsafe { *self.id_to_byte.get_unchecked(right as usize) };

            if b1 != 0xFF && b2 != 0xFF {
                let flat_idx = ((b1 as usize) << 8) | (b2 as usize);
                return unsafe { *self.byte_pair_ranks.get_unchecked(flat_idx) };
            }
        }

        let pack = ((left as u64) << 32) | (right as u64);
        let mut h = pack.wrapping_mul(0x517cc1b727220a95);
        h ^= h >> 47;

        let mask = self.hash_mask;
        let mut idx = (h & mask) as usize;

        unsafe {
            let keys_ptr = self.keys_flat.as_ptr();
            let values_ptr = self.values_flat.as_ptr();
            let table_mask = mask as usize;

            loop {
                let key = *keys_ptr.add(idx);
                if key == pack {
                    return *values_ptr.add(idx);
                }
                if key == u64::MAX {
                    return u64::MAX;
                }
                idx = (idx + 1) & table_mask;
            }
        }
    }
}

unsafe impl Send for BpeTokenizer {}
unsafe impl Sync for BpeTokenizer {}
