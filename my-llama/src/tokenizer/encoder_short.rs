use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

impl BpeTokenizer {
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
        let prev = unsafe { ctx.short_prev.get_unchecked_mut(..len) };
        let next = unsafe { ctx.short_next.get_unchecked_mut(..len) };
        let ids = unsafe { ctx.short_token_ids.get_unchecked_mut(..len) };

        for i in 0..len {
            unsafe {
                *prev.get_unchecked_mut(i) = i.wrapping_sub(1) as u8;
                *next.get_unchecked_mut(i) = (i + 1) as u8;
                *ids.get_unchecked_mut(i) = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize);
            }
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = usize::MAX;
            let mut best_id = 0u32;

            let mut i = 0;
            while i < len {
                let r = unsafe { *next.get_unchecked(i) as usize };
                if r >= len {
                    break;
                }

                let id_l = unsafe { *ids.get_unchecked(i) };
                let id_r = unsafe { *ids.get_unchecked(r) };

                if let Some(val) = self.get_pair_value(id_l, id_r) {
                    if val.rank < min_rank {
                        min_rank = val.rank;
                        best_id = val.id;
                        best_left = i;
                    }
                }
                i = r;
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
                *ids.get_unchecked_mut(l) = best_id;
            }
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            unsafe {
                *slice.get_unchecked_mut(*token_count) = *ids.get_unchecked(i);
                *token_count += 1;
                i = *next.get_unchecked(i) as usize;
            }
        }
    }
}
