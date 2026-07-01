use super::context::TokenizationContext;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

#[derive(Eq, PartialEq)]
pub struct BpePair {
    pub rank: u32,
    pub left_idx: usize,
}

impl Ord for BpePair {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank.cmp(&self.rank).then_with(|| self.left_idx.cmp(&other.left_idx))
    }
}

impl PartialOrd for BpePair {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct BpeTokenizer {
    pair_ranks: FxHashMap<u64, u32>,
    byte_pair_ranks: Box<[[u32; 256]; 256]>,
    byte_fallback: [u32; 256],
}

impl BpeTokenizer {
    pub fn new(pair_ranks: FxHashMap<u64, u32>, byte_fallback: [u32; 256]) -> Self {
        let mut byte_pair_ranks = Box::new([[u32::MAX; 256]; 256]);
        for b1 in 0..=255 {
            for b2 in 0..=255 {
                let id1 = byte_fallback[b1];
                let id2 = byte_fallback[b2];
                let pack = ((id1 as u64) << 32) | (id2 as u64);
                if let Some(&rank) = pair_ranks.get(&pack) {
                    byte_pair_ranks[b1][b2] = rank;
                }
            }
        }

        Self {
            pair_ranks,
            byte_pair_ranks,
            byte_fallback,
        }
    }

    #[inline(always)]
    fn get_pair_rank(&self, left: u32, right: u32) -> Option<u32> {
        if left < 256 && right < 256 {
            let rank = self.byte_pair_ranks[left as usize][right as usize];
            if rank == u32::MAX { None } else { Some(rank) }
        } else {
            let pack = ((left as u64) << 32) | (right as u64);
            self.pair_ranks.get(&pack).copied()
        }
    }

    pub fn encode_single_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        if len == 1 {
            if *token_count < slice.len() {
                slice[*token_count] = self.byte_fallback[bytes[0] as usize];
                *token_count += 1;
            }
            return;
        }

        if len <= 16 {
            self.encode_short_chunk(bytes, slice, token_count, ctx);
        } else {
            self.encode_long_chunk(bytes, slice, token_count, ctx);
        }
    }

    /// 1. КОРОТКИЙ ПУТЬ (Длина <= 16): Без кучи, линейный поиск на стеке.
    fn encode_short_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        let prev = &mut ctx.short_prev[..len];
        let next = &mut ctx.short_next[..len];
        let ids = &mut ctx.short_token_ids[..len];

        for i in 0..len {
            prev[i] = i.wrapping_sub(1) as u8;
            next[i] = (i + 1) as u8;
            ids[i] = self.byte_fallback[bytes[i] as usize];
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = usize::MAX;

            let mut i = 0;
            while i < len {
                let r = next[i] as usize;
                if r >= len {
                    break;
                }

                if let Some(rank) = self.get_pair_rank(ids[i], ids[r]) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_left = i;
                    }
                }
                i = r;
            }

            if best_left == usize::MAX {
                break;
            }

            let l = best_left;
            let r = next[l] as usize;
            let after_r = next[r];

            next[l] = after_r;
            if (after_r as usize) < len {
                prev[after_r as usize] = l as u8;
            }

            ids[l] = min_rank;
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            slice[*token_count] = ids[i];
            *token_count += 1;
            i = next[i] as usize;
        }
    }

    fn encode_long_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        ctx.prev.resize(len, 0);
        ctx.next.resize(len, 0);
        ctx.token_ids.resize(len, 0);
        ctx.heap.clear();

        for i in 0..len {
            ctx.prev[i] = i.wrapping_sub(1);
            ctx.next[i] = i + 1;
            ctx.token_ids[i] = self.byte_fallback[bytes[i] as usize];
        }

        for i in 0..len - 1 {
            if let Some(rank) = self.get_pair_rank(ctx.token_ids[i], ctx.token_ids[i + 1]) {
                ctx.heap.push(BpePair { rank, left_idx: i });
            }
        }

        while let Some(BpePair { rank, left_idx }) = ctx.heap.pop() {
            let r = ctx.next[left_idx];
            if r >= len {
                continue;
            }

            if self.get_pair_rank(ctx.token_ids[left_idx], ctx.token_ids[r]) != Some(rank) {
                continue;
            }

            let l_prev = ctx.prev[left_idx];
            let after_r = ctx.next[r];

            ctx.next[left_idx] = after_r;
            if after_r < len {
                ctx.prev[after_r] = left_idx;
            }

            ctx.token_ids[left_idx] = rank;

            if l_prev != usize::MAX {
                if let Some(r_new) = self.get_pair_rank(ctx.token_ids[l_prev], ctx.token_ids[left_idx]) {
                    ctx.heap.push(BpePair {
                        rank: r_new,
                        left_idx: l_prev,
                    });
                }
            }

            if after_r < len {
                if let Some(r_new) = self.get_pair_rank(ctx.token_ids[left_idx], ctx.token_ids[after_r]) {
                    ctx.heap.push(BpePair {
                        rank: r_new,
                        left_idx: left_idx,
                    });
                }
            }
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            slice[*token_count] = ctx.token_ids[i];
            *token_count += 1;
            i = ctx.next[i];
        }
    }
}
