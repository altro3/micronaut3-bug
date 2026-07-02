use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

macro_rules! generate_bpe_loop {
    ($N:expr, $self:expr, $len:expr, $cached_ranks:expr, $cached_ids:expr, $prev:expr, $next:expr, $ids:expr, $token_count:expr, $slice:expr, $slice_len:expr) => {
        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = usize::MAX;

            for i in 0..($N - 1) {
                let rk = unsafe { *$cached_ranks.get_unchecked(i) };
                if rk < min_rank {
                    min_rank = rk;
                    best_left = i;
                }
            }

            if best_left == usize::MAX {
                break;
            }

            let l = best_left;
            let r = unsafe { *$next.get_unchecked(l) as usize };
            let after_r = unsafe { *$next.get_unchecked(r) };

            unsafe {
                *$next.get_unchecked_mut(l) = after_r;
                if (after_r as usize) < $len {
                    *$prev.get_unchecked_mut(after_r as usize) = l as u8;
                }
                *$ids.get_unchecked_mut(l) = *$cached_ids.get_unchecked(l);
            }

            unsafe {
                *$cached_ranks.get_unchecked_mut(l) = u32::MAX;
                *$cached_ranks.get_unchecked_mut(r) = u32::MAX;
            }

            if (after_r as usize) < $len {
                let id_l = unsafe { *$ids.get_unchecked(l) };
                let id_after = unsafe { *$ids.get_unchecked(after_r as usize) };

                let packed = $self.get_pair_packed(id_l, id_after);
                if packed != u64::MAX {
                    unsafe {
                        *$cached_ranks.get_unchecked_mut(l) = (packed >> 32) as u32;
                        *$cached_ids.get_unchecked_mut(l) = packed as u32;
                    }
                }
            }

            let before_l = unsafe { *$prev.get_unchecked(l) as usize };
            if before_l < $len {
                let id_before = unsafe { *$ids.get_unchecked(before_l) };
                let id_l = unsafe { *$ids.get_unchecked(l) };

                let packed = $self.get_pair_packed(id_before, id_l);
                if packed != u64::MAX {
                    unsafe {
                        *$cached_ranks.get_unchecked_mut(before_l) = (packed >> 32) as u32;
                        *$cached_ids.get_unchecked_mut(before_l) = packed as u32;
                    }
                } else {
                    unsafe {
                        *$cached_ranks.get_unchecked_mut(before_l) = u32::MAX;
                    }
                }
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

        let mut count = *token_count;
        let slice_len = slice.len();

        match len {
            2 => {
                generate_bpe_loop!(2, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            3 => {
                generate_bpe_loop!(3, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            4 => {
                generate_bpe_loop!(4, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            5 => {
                generate_bpe_loop!(5, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            6 => {
                generate_bpe_loop!(6, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            7 => {
                generate_bpe_loop!(7, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            8 => {
                generate_bpe_loop!(8, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            9 => {
                generate_bpe_loop!(9, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            10 => {
                generate_bpe_loop!(10, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            11 => {
                generate_bpe_loop!(11, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            12 => {
                generate_bpe_loop!(12, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            13 => {
                generate_bpe_loop!(13, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            14 => {
                generate_bpe_loop!(14, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            15 => {
                generate_bpe_loop!(15, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            16 => {
                generate_bpe_loop!(16, self, len, cached_ranks, cached_ids, prev, next, ids, count, slice, slice_len);
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
        }

        let mut i = 0;
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
