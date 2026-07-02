use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

impl BpeTokenizer {
    pub(crate) fn encode_long_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();

        if len > ctx.heap.next_node_len() {
            ctx.heap.reserve_chunk_len(len + 256);
        }

        ctx.prev.resize(len * 3, 0);
        ctx.heap.clear();

        for i in 0..len {
            let base = i * 3;
            unsafe {
                *ctx.prev.get_unchecked_mut(base) = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize) as usize;
                *ctx.prev.get_unchecked_mut(base + 1) = i.wrapping_sub(1);
                *ctx.prev.get_unchecked_mut(base + 2) = i + 1;
            }
        }

        for i in 0..len - 1 {
            unsafe {
                let id1 = *ctx.prev.get_unchecked(i * 3) as u32;
                let id2 = *ctx.prev.get_unchecked((i + 1) * 3) as u32;
                if let Some(val) = self.get_pair_value(id1, id2) {
                    ctx.heap.push(val.rank, i);
                }
            }
        }

        while let Some(pair) = ctx.heap.pop() {
            let left_idx = pair.left_idx;
            let rank = pair.rank;
            let base_l = left_idx * 3;

            if base_l + 2 >= ctx.prev.len() {
                continue;
            }

            let r = unsafe { *ctx.prev.get_unchecked(base_l + 2) };
            if r >= len {
                continue;
            }
            let base_r = r * 3;
            if base_r >= ctx.prev.len() {
                continue;
            }

            let id_l = unsafe { *ctx.prev.get_unchecked(base_l) as u32 };
            let id_r = unsafe { *ctx.prev.get_unchecked(base_r) as u32 };

            let current_val = self.get_pair_value(id_l, id_r);
            if current_val.map(|v| v.rank) != Some(rank) {
                continue;
            }

            let target_id = current_val.unwrap().id;
            let l_prev = unsafe { *ctx.prev.get_unchecked(base_l + 1) };
            let after_r = unsafe { *ctx.prev.get_unchecked(base_r + 2) };

            unsafe {
                *ctx.prev.get_unchecked_mut(base_l + 2) = after_r;
                if after_r < len {
                    *ctx.prev.get_unchecked_mut(after_r * 3 + 1) = left_idx;
                }
                *ctx.prev.get_unchecked_mut(base_l) = target_id as usize;
            }

            if l_prev != usize::MAX && l_prev * 3 < ctx.prev.len() {
                unsafe {
                    let id_l_prev = *ctx.prev.get_unchecked(l_prev * 3) as u32;
                    if let Some(val) = self.get_pair_value(id_l_prev, target_id) {
                        ctx.heap.push(val.rank, l_prev);
                    }
                }
            }

            if after_r < len && after_r * 3 < ctx.prev.len() {
                unsafe {
                    let id_after_r = *ctx.prev.get_unchecked(after_r * 3) as u32;
                    if let Some(val) = self.get_pair_value(target_id, id_after_r) {
                        ctx.heap.push(val.rank, left_idx);
                    }
                }
            }
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            let base = i * 3;
            if base >= ctx.prev.len() {
                break;
            }
            unsafe {
                *slice.get_unchecked_mut(*token_count) = *ctx.prev.get_unchecked(base) as u32;
                *token_count += 1;
                i = *ctx.prev.get_unchecked(base + 2);
            }
        }
    }
}
