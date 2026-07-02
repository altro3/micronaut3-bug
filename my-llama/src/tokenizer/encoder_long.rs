use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

impl BpeTokenizer {
    pub(crate) fn encode_long_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();

        if len > ctx.heap.next_node_len() {
            ctx.heap.reserve_chunk_len(len + 256);
        }

        if ctx.long_ids.capacity() < len {
            let additional = len - ctx.long_ids.len();
            ctx.long_ids.reserve(additional);
            ctx.long_prev.reserve(additional);
            ctx.long_next.reserve(additional);
        }

        unsafe {
            ctx.long_ids.set_len(len);
            ctx.long_prev.set_len(len);
            ctx.long_next.set_len(len);
        }
        ctx.heap.clear(len);

        for i in 0..len {
            unsafe {
                let id = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize);
                *ctx.long_ids.get_unchecked_mut(i) = id;
                *ctx.long_prev.get_unchecked_mut(i) = if i == 0 { -1 } else { (i - 1) as i32 };
                *ctx.long_next.get_unchecked_mut(i) = if i == len - 1 { -1 } else { (i + 1) as i32 };
            }
        }

        for i in 0..len - 1 {
            unsafe {
                let id1 = *ctx.long_ids.get_unchecked(i);
                let id2 = *ctx.long_ids.get_unchecked(i + 1);

                let packed = self.get_pair_packed(id1, id2);
                if packed != u64::MAX {
                    let rank = (packed >> 32) as u32;
                    ctx.heap.push(rank, i);
                }
            }
        }

        loop {
            let packed_pair = ctx.heap.pop_packed();
            if packed_pair == u64::MAX {
                break;
            }

            let rank = (packed_pair >> 32) as u32;
            let l = packed_pair as u32 as usize;

            let r = unsafe { *ctx.long_next.get_unchecked(l) };
            if r == -1 {
                continue;
            }
            let r_idx = r as usize;

            unsafe {
                let id_l = *ctx.long_ids.get_unchecked(l);
                let id_r = *ctx.long_ids.get_unchecked(r_idx);

                let packed = self.get_pair_packed(id_l, id_r);
                if packed == u64::MAX || (packed >> 32) as u32 != rank {
                    continue;
                }

                let target_id = packed as u32;
                let l_prev = *ctx.long_prev.get_unchecked(l);
                let after_r = *ctx.long_next.get_unchecked(r_idx);

                *ctx.long_next.get_unchecked_mut(l) = after_r;
                if after_r != -1 {
                    *ctx.long_prev.get_unchecked_mut(after_r as usize) = l as i32;
                }
                *ctx.long_ids.get_unchecked_mut(l) = target_id;

                *ctx.long_next.get_unchecked_mut(r_idx) = -1;
                *ctx.long_prev.get_unchecked_mut(r_idx) = -1;
                *ctx.long_ids.get_unchecked_mut(r_idx) = u32::MAX;

                if l_prev != -1 {
                    let l_prev_idx = l_prev as usize;
                    let id_prev = *ctx.long_ids.get_unchecked(l_prev_idx);
                    let packed_l = self.get_pair_packed(id_prev, target_id);
                    if packed_l != u64::MAX {
                        ctx.heap.push((packed_l >> 32) as u32, l_prev_idx);
                    }
                }

                if after_r != -1 {
                    let after_r_idx = after_r as usize;
                    let id_after = *ctx.long_ids.get_unchecked(after_r_idx);
                    let packed_r = self.get_pair_packed(target_id, id_after);
                    if packed_r != u64::MAX {
                        ctx.heap.push((packed_r >> 32) as u32, l);
                    }
                }
            }
        }

        let mut i = 0usize;
        let mut count = *token_count;
        let slice_len = slice.len();

        while i < len {
            if count >= slice_len {
                break;
            }
            unsafe {
                *slice.get_unchecked_mut(count) = *ctx.long_ids.get_unchecked(i);
                count += 1;

                let next_node = *ctx.long_next.get_unchecked(i);
                if next_node == -1 {
                    break;
                }
                i = next_node as usize;
            }
        }
        *token_count = count;
    }
}
