use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

macro_rules! find_min_rank {
    ($N:expr, $cached_ranks:expr, $min_rank:expr, $best_left:expr) => {
        for i in 0..$N {
            let rk = unsafe { *$cached_ranks.get_unchecked(i) };
            if rk < $min_rank {
                $min_rank = rk;
                $best_left = i;
            }
        }
    };
}

impl BpeTokenizer {
    #[inline(always)]
    pub fn encode_single_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        if len == 1 {
            if *token_count < slice.len() {
                unsafe {
                    *slice.get_unchecked_mut(*token_count) = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(0) as usize);
                }
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

    fn encode_short_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len > 16 {
            return;
        }

        let mut cached_ranks = [u32::MAX; 16];
        let mut cached_ids = [0u32; 16];

        let prev = unsafe { ctx.short_prev.get_unchecked_mut(..len) };
        let next = unsafe { ctx.short_next.get_unchecked_mut(..len) };
        let ids = unsafe { ctx.short_token_ids.get_unchecked_mut(..len) };

        for i in 0..len {
            unsafe {
                let id = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize);
                *ids.get_unchecked_mut(i) = id;
                *prev.get_unchecked_mut(i) = if i == 0 { 0xFF } else { (i - 1) as u8 };
                *next.get_unchecked_mut(i) = (i + 1) as u8;
            }
        }

        for i in 0..len - 1 {
            let id_l = unsafe { *ids.get_unchecked(i) };
            let id_r = unsafe { *ids.get_unchecked(i + 1) };

            let packed = self.get_pair_packed(id_l, id_r);
            if packed != u64::MAX {
                unsafe {
                    *cached_ranks.get_unchecked_mut(i) = (packed >> 32) as u32;
                    *cached_ids.get_unchecked_mut(i) = packed as u32;
                }
            }
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = usize::MAX;

            match len {
                2 => find_min_rank!(2, cached_ranks, min_rank, best_left),
                3 => find_min_rank!(3, cached_ranks, min_rank, best_left),
                4 => find_min_rank!(4, cached_ranks, min_rank, best_left),
                5 => find_min_rank!(5, cached_ranks, min_rank, best_left),
                6 => find_min_rank!(6, cached_ranks, min_rank, best_left),
                7 => find_min_rank!(7, cached_ranks, min_rank, best_left),
                8 => find_min_rank!(8, cached_ranks, min_rank, best_left),
                9 => find_min_rank!(9, cached_ranks, min_rank, best_left),
                10 => find_min_rank!(10, cached_ranks, min_rank, best_left),
                11 => find_min_rank!(11, cached_ranks, min_rank, best_left),
                12 => find_min_rank!(12, cached_ranks, min_rank, best_left),
                13 => find_min_rank!(13, cached_ranks, min_rank, best_left),
                14 => find_min_rank!(14, cached_ranks, min_rank, best_left),
                15 => find_min_rank!(15, cached_ranks, min_rank, best_left),
                16 => find_min_rank!(16, cached_ranks, min_rank, best_left),
                _ => unsafe { std::hint::unreachable_unchecked() },
            }

            if best_left == usize::MAX {
                break;
            }

            let l = best_left;
            let r = unsafe { *next.get_unchecked(l) as usize };
            let after_r = unsafe { *next.get_unchecked(r) };

            unsafe {
                *next.get_unchecked_mut(l) = after_r;
                if (after_r as usize) < len {
                    *prev.get_unchecked_mut(after_r as usize) = l as u8;
                }
                *ids.get_unchecked_mut(l) = *cached_ids.get_unchecked(l);
            }

            unsafe {
                *cached_ranks.get_unchecked_mut(l) = u32::MAX;
                *cached_ranks.get_unchecked_mut(r) = u32::MAX;
            }

            if (after_r as usize) < len {
                let id_l = unsafe { *ids.get_unchecked(l) };
                let id_after = unsafe { *ids.get_unchecked(after_r as usize) };

                let packed = self.get_pair_packed(id_l, id_after);
                if packed != u64::MAX {
                    unsafe {
                        *cached_ranks.get_unchecked_mut(l) = (packed >> 32) as u32;
                        *cached_ids.get_unchecked_mut(l) = packed as u32;
                    }
                }
            }

            let before_l = unsafe { *prev.get_unchecked(l) as usize };
            if before_l < len {
                let id_before = unsafe { *ids.get_unchecked(before_l) };
                let id_l = unsafe { *ids.get_unchecked(l) };

                let packed = self.get_pair_packed(id_before, id_l);
                if packed != u64::MAX {
                    unsafe {
                        *cached_ranks.get_unchecked_mut(before_l) = (packed >> 32) as u32;
                        *cached_ids.get_unchecked_mut(before_l) = packed as u32;
                    }
                } else {
                    unsafe {
                        *cached_ranks.get_unchecked_mut(before_l) = u32::MAX;
                    }
                }
            }
        }

        let mut i = 0;
        let mut count = *token_count;
        let slice_len = slice.len();

        while i < len {
            if count >= slice_len {
                break;
            }
            unsafe {
                *slice.get_unchecked_mut(count) = *ids.get_unchecked(i);
                count += 1;
                i = *next.get_unchecked(i) as usize;
            }
        }
        *token_count = count;
    }
}
