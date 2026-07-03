use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

impl BpeTokenizer {
    pub(crate) fn encode_long_chunk(&self, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        if len > ctx.long_ids.capacity() {
            let new_cap = len.next_power_of_two();
            ctx.long_ids.reserve_exact(new_cap - ctx.long_ids.len());
            ctx.long_prev.reserve_exact(new_cap - ctx.long_prev.len());
            ctx.long_next.reserve_exact(new_cap - ctx.long_next.len());
        }

        unsafe {
            ctx.long_ids.set_len(len);
            ctx.long_prev.set_len(len);
            ctx.long_next.set_len(len);
        }

        if len > ctx.heap.next_node.len() {
            ctx.heap.next_node.resize(len + 256, u32::MAX);
        }

        if len > ctx.long_ranks.capacity() {
            ctx.long_ranks.reserve_exact(len.next_power_of_two() - ctx.long_ranks.len());
        }
        unsafe {
            ctx.long_ranks.set_len(len);
            std::ptr::write_bytes(ctx.long_ranks.as_mut_ptr(), 0xFF, len);
        }

        ctx.heap.clear(len);

        let long_ids_ptr = ctx.long_ids.as_mut_ptr();
        let long_prev_ptr = ctx.long_prev.as_mut_ptr();
        let long_next_ptr = ctx.long_next.as_mut_ptr();
        let long_ranks_ptr = ctx.long_ranks.as_mut_ptr();
        let fallback_ptr = self.byte_fallback.as_ptr();

        let mut idx = 0;
        unsafe {
            while idx + 4 <= len {
                let b0 = *bytes.get_unchecked(idx) as usize;
                let b1 = *bytes.get_unchecked(idx + 1) as usize;
                let b2 = *bytes.get_unchecked(idx + 2) as usize;
                let b3 = *bytes.get_unchecked(idx + 3) as usize;

                *long_ids_ptr.add(idx) = *fallback_ptr.add(b0);
                *long_ids_ptr.add(idx + 1) = *fallback_ptr.add(b1);
                *long_ids_ptr.add(idx + 2) = *fallback_ptr.add(b2);
                *long_ids_ptr.add(idx + 3) = *fallback_ptr.add(b3);

                *long_prev_ptr.add(idx) = idx as i32 - 1;
                *long_next_ptr.add(idx) = idx as i32 + 1;

                *long_prev_ptr.add(idx + 1) = idx as i32;
                *long_next_ptr.add(idx + 1) = idx as i32 + 2;

                *long_prev_ptr.add(idx + 2) = idx as i32 + 1;
                *long_next_ptr.add(idx + 2) = idx as i32 + 3;

                *long_prev_ptr.add(idx + 3) = idx as i32 + 2;
                *long_next_ptr.add(idx + 3) = idx as i32 + 4;

                idx += 4;
            }
            while idx < len {
                let b = *bytes.get_unchecked(idx) as usize;
                *long_ids_ptr.add(idx) = *fallback_ptr.add(b);
                *long_prev_ptr.add(idx) = idx as i32 - 1;
                *long_next_ptr.add(idx) = idx as i32 + 1;
                idx += 1;
            }
            *long_next_ptr.add(len - 1) = -1;
        }

        for i in 0..(len - 1) {
            unsafe {
                let id1 = *long_ids_ptr.add(i);
                let id2 = *long_ids_ptr.add(i + 1);

                let packed = self.get_pair_packed(id1, id2);
                if packed != u64::MAX {
                    let r = (packed >> 32) as u32;
                    *long_ranks_ptr.add(i) = r;
                    ctx.heap.push(r, i);
                }
            }
        }

        loop {
            let packed_pair = ctx.heap.pop_packed();
            if packed_pair == u64::MAX {
                break;
            }

            let rank = (packed_pair >> 32) as u32;
            let l = (packed_pair as u32) as usize;

            unsafe {
                if *long_ranks_ptr.add(l) != rank {
                    continue;
                }

                let r = *long_next_ptr.add(l);
                if r == -1 {
                    continue;
                }
                let r_idx = r as usize;

                let id_l = *long_ids_ptr.add(l);
                let id_r = *long_ids_ptr.add(r_idx);

                let packed = self.get_pair_packed(id_l, id_r);
                if packed == u64::MAX || (packed >> 32) as u32 != rank {
                    continue;
                }

                let target_id = packed as u32;
                let l_prev = *long_prev_ptr.add(l);
                let after_r = *long_next_ptr.add(r_idx);

                *long_next_ptr.add(l) = after_r;
                if after_r != -1 {
                    *long_prev_ptr.add(after_r as usize) = l as i32;
                }
                *long_ids_ptr.add(l) = target_id;

                *long_ranks_ptr.add(l) = u32::MAX;
                *long_ranks_ptr.add(r_idx) = u32::MAX;
                *long_next_ptr.add(r_idx) = -1;
                *long_prev_ptr.add(r_idx) = -1;

                if l_prev != -1 {
                    let l_prev_idx = l_prev as usize;
                    let id_prev = *long_ids_ptr.add(l_prev_idx);
                    let packed_l = self.get_pair_packed(id_prev, target_id);
                    if packed_l != u64::MAX {
                        let r_l = (packed_l >> 32) as u32;
                        *long_ranks_ptr.add(l_prev_idx) = r_l;
                        ctx.heap.push(r_l, l_prev_idx);
                    } else {
                        *long_ranks_ptr.add(l_prev_idx) = u32::MAX;
                    }
                }

                if after_r != -1 {
                    let after_r_idx = after_r as usize;
                    let id_after = *long_ids_ptr.add(after_r_idx);
                    let packed_r = self.get_pair_packed(target_id, id_after);
                    if packed_r != u64::MAX {
                        let r_r = (packed_r >> 32) as u32;
                        *long_ranks_ptr.add(l) = r_r;
                        ctx.heap.push(r_r, l);
                    } else {
                        *long_ranks_ptr.add(l) = u32::MAX;
                    }
                }
            }
        }

        let mut i = 0i32;
        let mut count = *token_count;

        let required_capacity = count + len;
        if required_capacity > ctx.tokens_buffer.capacity() {
            ctx.tokens_buffer.reserve(required_capacity - ctx.tokens_buffer.len());
        }

        let out_tokens_ptr = ctx.tokens_buffer.as_mut_ptr();
        unsafe {
            while i != -1 {
                let idx = i as usize;
                *out_tokens_ptr.add(count) = *long_ids_ptr.add(idx);
                count += 1;
                i = *long_next_ptr.add(idx);
            }
            ctx.tokens_buffer.set_len(count);
        }

        *token_count = count;
    }
}
