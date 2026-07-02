use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

impl BpeTokenizer {
    pub(crate) fn encode_long_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
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

        let mut min_rank = ctx.heap.buckets.len();
        let mut max_rank_dirty = ctx.heap.buckets.len() - 1;

        let (n_min, n_max) = ctx.heap.clear(min_rank, max_rank_dirty, len);
        min_rank = n_min;
        max_rank_dirty = n_max;

        let fallback_ptr = self.byte_fallback.as_ptr();
        let bytes_ptr = bytes.as_ptr();

        let long_ids_ptr = ctx.long_ids.as_mut_ptr();
        let long_prev_ptr = ctx.long_prev.as_mut_ptr();
        let long_next_ptr = ctx.long_next.as_mut_ptr();

        for i in 0..len {
            unsafe {
                let id = *fallback_ptr.add(*bytes_ptr.add(i) as usize);
                std::ptr::write(long_ids_ptr.add(i), id);
                std::ptr::write(long_prev_ptr.add(i), if i == 0 { -1 } else { (i - 1) as i32 });
                std::ptr::write(long_next_ptr.add(i), if i == len - 1 { -1 } else { (i + 1) as i32 });
            }
        }

        for i in 0..len - 1 {
            unsafe {
                let id1 = *long_ids_ptr.add(i);
                let id2 = *long_ids_ptr.add(i + 1);

                let packed = self.get_pair_packed(id1, id2);
                if packed != u64::MAX {
                    let rank = (packed >> 32) as u32;
                    let (n_min, n_max) = ctx.heap.push(min_rank, max_rank_dirty, rank, i);
                    min_rank = n_min;
                    max_rank_dirty = n_max;
                }
            }
        }

        loop {
            let (packed_pair, new_min) = ctx.heap.pop_packed(min_rank);
            min_rank = new_min;

            if packed_pair == u64::MAX {
                break;
            }

            let rank = (packed_pair >> 32) as u32;
            let l = packed_pair as u32 as usize;

            unsafe {
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

                *long_next_ptr.add(r_idx) = -1;
                *long_prev_ptr.add(r_idx) = -1;
                *long_ids_ptr.add(r_idx) = u32::MAX;

                if l_prev != -1 {
                    let l_prev_idx = l_prev as usize;
                    let id_prev = *long_ids_ptr.add(l_prev_idx);
                    let packed_l = self.get_pair_packed(id_prev, target_id);
                    if packed_l != u64::MAX {
                        let (n_min, n_max) = ctx.heap.push(min_rank, max_rank_dirty, (packed_l >> 32) as u32, l_prev_idx);
                        min_rank = n_min;
                        max_rank_dirty = n_max;
                    }
                }

                if after_r != -1 {
                    let after_r_idx = after_r as usize;
                    let id_after = *long_ids_ptr.add(after_r_idx);
                    let packed_r = self.get_pair_packed(target_id, id_after);
                    if packed_r != u64::MAX {
                        let (n_min, n_max) = ctx.heap.push(min_rank, max_rank_dirty, (packed_r >> 32) as u32, l);
                        min_rank = n_min;
                        max_rank_dirty = n_max;
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
                *slice.get_unchecked_mut(count) = *long_ids_ptr.add(i);
                count += 1;

                let next_node = *long_next_ptr.add(i);
                if next_node == -1 {
                    break;
                }
                i = next_node as usize;
            }
        }
        *token_count = count;
    }
}
