use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

impl BpeTokenizer {
    pub(crate) fn encode_long_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();

        if len > ctx.heap.next_node_len() {
            ctx.heap.reserve_chunk_len(len + 256);
        }

        let required_slots = len * 3;
        if ctx.prev.capacity() < required_slots {
            ctx.prev.reserve(required_slots - ctx.prev.len());
        }
        unsafe {
            ctx.prev.set_len(required_slots);
        }
        ctx.heap.clear();

        let boundary_terminator = 0x7FFFFFFF_FFFFFFFFusize;

        for i in 0..len {
            let base = i * 3;
            unsafe {
                let id = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize) as usize;
                *ctx.prev.get_unchecked_mut(base) = id;
                *ctx.prev.get_unchecked_mut(base + 1) = if i == 0 { boundary_terminator } else { i - 1 };
                *ctx.prev.get_unchecked_mut(base + 2) = i + 1;
            }
        }

        for i in 0..len - 1 {
            unsafe {
                let id1 = *ctx.prev.get_unchecked(i * 3) as u32;
                let id2 = *ctx.prev.get_unchecked((i + 1) * 3) as u32;

                let packed = self.get_pair_packed(id1, id2);
                if packed != u64::MAX {
                    let rank = (packed >> 32) as u32;
                    ctx.heap.push(rank, i);
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

            let packed = self.get_pair_packed(id_l, id_r);
            if packed == u64::MAX || (packed >> 32) as u32 != rank {
                continue;
            }

            let target_id = packed as u32;
            let l_prev = unsafe { *ctx.prev.get_unchecked(base_l + 1) };
            let after_r = unsafe { *ctx.prev.get_unchecked(base_r + 2) };

            unsafe {
                *ctx.prev.get_unchecked_mut(base_l + 2) = after_r;
                if after_r < len {
                    *ctx.prev.get_unchecked_mut(after_r * 3 + 1) = left_idx;
                }
                *ctx.prev.get_unchecked_mut(base_l) = target_id as usize;
            }

            if l_prev != boundary_terminator && l_prev * 3 < ctx.prev.len() {
                unsafe {
                    let id_l_prev = *ctx.prev.get_unchecked(l_prev * 3) as u32;
                    let packed_l = self.get_pair_packed(id_l_prev, target_id);
                    if packed_l != u64::MAX {
                        ctx.heap.push((packed_l >> 32) as u32, l_prev);
                    }
                }
            }

            if after_r < len && after_r * 3 < ctx.prev.len() {
                unsafe {
                    let id_after_r = *ctx.prev.get_unchecked(after_r * 3) as u32;
                    let packed_r = self.get_pair_packed(target_id, id_after_r);
                    if packed_r != u64::MAX {
                        ctx.heap.push((packed_r >> 32) as u32, left_idx);
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
            let base = i * 3;
            if base >= ctx.prev.len() {
                break;
            }
            unsafe {
                *slice.get_unchecked_mut(count) = *ctx.prev.get_unchecked(base) as u32;
                count += 1;
                i = *ctx.prev.get_unchecked(base + 2);
            }
        }
        *token_count = count;
    }
}
