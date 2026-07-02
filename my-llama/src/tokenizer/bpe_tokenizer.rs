use super::bpe_types::BpeValue;
use rustc_hash::FxHashMap;

pub struct BpeTokenizer {
    pub(crate) pair_ranks: FxHashMap<u64, BpeValue>,
    pub(crate) byte_pair_ranks: Box<[[u64; 256]; 256]>,
    pub(crate) byte_fallback: [u32; 256],
    pub(crate) id_to_byte: [i16; 513],
    pub eos_token_id: u32,
    pub(crate) vocab_size: usize,
}

impl BpeTokenizer {
    pub fn new(pair_ranks: FxHashMap<u64, BpeValue>, byte_fallback: [u32; 256], eos_token_id: u32) -> Self {
        let mut byte_pair_ranks = Box::new([[u64::MAX; 256]; 256]);
        let mut id_to_byte = [-1i16; 513];

        let vocab_size = (pair_ranks.len() * 2).max(160000);

        for b in 0..=255 {
            let id = byte_fallback[b] as usize;
            if id < 512 {
                id_to_byte[id] = b as i16;
            }
        }

        for b1 in 0..=255 {
            for b2 in 0..=255 {
                let id1 = byte_fallback[b1];
                let id2 = byte_fallback[b2];
                let pack = ((id1 as u64) << 32) | (id2 as u64);
                if let Some(&val) = pair_ranks.get(&pack) {
                    byte_pair_ranks[b1][b2] = ((val.rank as u64) << 32) | (val.id as u64);
                }
            }
        }

        Self {
            pair_ranks,
            byte_pair_ranks,
            byte_fallback,
            id_to_byte,
            eos_token_id,
            vocab_size,
        }
    }

    #[inline(always)]
    pub(crate) fn get_pair_value(&self, left: u32, right: u32) -> Option<BpeValue> {
        let idx_left = if left < 512 { left as usize } else { 512 };
        let idx_right = if right < 512 { right as usize } else { 512 };

        let b1 = unsafe { *self.id_to_byte.get_unchecked(idx_left) };
        let b2 = unsafe { *self.id_to_byte.get_unchecked(idx_right) };

        if (b1 >= 0) & (b2 >= 0) {
            let packed = unsafe { *self.byte_pair_ranks.get_unchecked(b1 as usize).get_unchecked(b2 as usize) };

            if packed != u64::MAX {
                return Some(BpeValue {
                    rank: (packed >> 32) as u32,
                    id: packed as u32,
                });
            }
        }

        let pack = ((left as u64) << 32) | (right as u64);
        self.pair_ranks.get(&pack).copied()
    }
}

unsafe impl Send for BpeTokenizer {}
unsafe impl Sync for BpeTokenizer {}
