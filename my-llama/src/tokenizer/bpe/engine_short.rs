use super::context::TokenizationContext;
use crate::tokenizer::BpeTokenizer;

pub struct ShortBpeEngine;

impl ShortBpeEngine {
    #[inline(always)]
    pub fn merge(data: &BpeTokenizer, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        let mut ids = [0u32; 32];
        let mut next = [0u16; 32];
        let mut prev = [0u16; 32];
        let mut ranks = [u32::MAX; 32];

        unsafe {
            let cache_ptr = ctx.bpe_direct_cache.as_mut_ptr();

            macro_rules! get_pair_cached {
                ($left:expr, $right:expr) => {{
                    let l = $left;
                    let r = $right;
                    let pack = ((l as u64) << 32) | (r as u64);

                    let cache_idx = ((l ^ r) & 0x3FFF) as usize;
                    let slot_ptr = cache_ptr.add(cache_idx);

                    if (*slot_ptr).key == pack {
                        (*slot_ptr).val
                    } else {
                        let res = data.get_pair_packed(l, r);
                        (*slot_ptr).key = pack;
                        (*slot_ptr).val = res;
                        res
                    }
                }};
            }

            let fallback_ptr = data.byte_fallback.as_ptr();
            for i in 0..len {
                let b = *bytes.get_unchecked(i) as usize;
                *ids.get_unchecked_mut(i) = *fallback_ptr.add(b);
                *next.get_unchecked_mut(i) = if i == len - 1 { 0xFFFF } else { (i + 1) as u16 };
                *prev.get_unchecked_mut(i) = if i == 0 { 0xFFFF } else { (i - 1) as u16 };
            }

            for i in 0..(len - 1) {
                let packed = get_pair_cached!(*ids.get_unchecked(i), *ids.get_unchecked(i + 1));
                *ranks.get_unchecked_mut(i) = if packed != u64::MAX { (packed >> 32) as u32 } else { u32::MAX };
            }

            loop {
                let (mut min_rank, mut best_left) = (u32::MAX, 0xFFFFusize);
                let mut i = 0usize;
                loop {
                    let r = *ranks.get_unchecked(i);
                    if r < min_rank {
                        min_rank = r;
                        best_left = i;
                    }
                    let n = *next.get_unchecked(i);
                    if n == 0xFFFF {
                        break;
                    }
                    i = n as usize;
                }

                if min_rank == u32::MAX {
                    break;
                }

                let right_idx = *next.get_unchecked(best_left) as usize;
                let packed = get_pair_cached!(*ids.get_unchecked(best_left), *ids.get_unchecked(right_idx));

                *ids.get_unchecked_mut(best_left) = packed as u32;
                let after_r = *next.get_unchecked(right_idx);
                *next.get_unchecked_mut(best_left) = after_r;

                if after_r != 0xFFFF {
                    *prev.get_unchecked_mut(after_r as usize) = best_left as u16;
                }
                *ranks.get_unchecked_mut(right_idx) = u32::MAX;

                if after_r != 0xFFFF {
                    let packed_r = get_pair_cached!(*ids.get_unchecked(best_left), *ids.get_unchecked(after_r as usize));
                    *ranks.get_unchecked_mut(best_left) = if packed_r != u64::MAX { (packed_r >> 32) as u32 } else { u32::MAX };
                } else {
                    *ranks.get_unchecked_mut(best_left) = u32::MAX;
                }

                let p_idx = *prev.get_unchecked(best_left);
                if p_idx != 0xFFFF {
                    let p = p_idx as usize;
                    let packed_l = get_pair_cached!(*ids.get_unchecked(p), *ids.get_unchecked(best_left));
                    *ranks.get_unchecked_mut(p) = if packed_l != u64::MAX { (packed_l >> 32) as u32 } else { u32::MAX };
                }
            }

            let mut curr = 0usize;
            let mut count = *token_count;
            let out_ptr = ctx.tokens_buffer.as_mut_ptr();
            loop {
                *out_ptr.add(count) = *ids.get_unchecked(curr);
                count += 1;
                let n = *next.get_unchecked(curr);
                if n == 0xFFFF {
                    break;
                }
                curr = n as usize;
            }
            *token_count = count;
            ctx.tokens_buffer.set_len(count);
        }
    }
}
