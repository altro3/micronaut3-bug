use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct ShortNode {
    rank: u32,
    id: u32,
    next: u8,
    prev: u8,
    _pad: [u8; 6],
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
            self.encode_short_chunk(bytes, slice, token_count);
        } else {
            self.encode_long_chunk(bytes, token_count, ctx);
        }
    }

    fn encode_short_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize) {
        let len = bytes.len();

        let mut nodes = [ShortNode {
            rank: u32::MAX,
            id: 0,
            next: 0xFF,
            prev: 0xFF,
            _pad: [0; 6],
        }; 16];
        let nodes_ptr = nodes.as_mut_ptr();

        for i in 0..len {
            unsafe {
                let id = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize);
                let node = nodes_ptr.add(i);
                (*node).id = id;
                (*node).prev = if i == 0 { 0xFF } else { (i - 1) as u8 };
                (*node).next = if i == len - 1 { 0xFF } else { (i + 1) as u8 };
            }
        }

        for i in 0..(len - 1) {
            unsafe {
                let id_l = (*nodes_ptr.add(i)).id;
                let id_r = (*nodes_ptr.add(i + 1)).id;
                let packed = self.get_pair_packed(id_l, id_r);
                if packed != u64::MAX {
                    (*nodes_ptr.add(i)).rank = (packed >> 32) as u32;
                    (*nodes_ptr.add(i)).id = packed as u32; // Сохраняем целевой ID токена
                }
            }
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = 0xFFu8;

            for i in 0..15 {
                unsafe {
                    let rk = (*nodes_ptr.add(i)).rank;
                    if rk < min_rank {
                        min_rank = rk;
                        best_left = i as u8;
                    }
                }
            }

            if min_rank == u32::MAX || best_left == 0xFF {
                break;
            }

            unsafe {
                let l_idx = best_left as usize;
                let node_l = nodes_ptr.add(l_idx);

                let r_idx = (*node_l).next as usize;
                let node_r = nodes_ptr.add(r_idx);

                let after_r_idx = (*node_r).next;

                // Сливаем узлы: l поглощает r
                (*node_l).next = after_r_idx;
                if after_r_idx != 0xFF {
                    (*nodes_ptr.add(after_r_idx as usize)).prev = best_left;
                }

                (*node_r).rank = u32::MAX;
                (*node_r).next = 0xFF;
                (*node_r).prev = 0xFF;

                if after_r_idx != 0xFF {
                    let id_l = (*node_l).id;
                    let id_after = (*nodes_ptr.add(after_r_idx as usize)).id;
                    let packed = self.get_pair_packed(id_l, id_after);
                    if packed != u64::MAX {
                        (*node_l).rank = (packed >> 32) as u32;
                        (*node_l).id = packed as u32;
                    } else {
                        (*node_l).rank = u32::MAX;
                    }
                } else {
                    (*node_l).rank = u32::MAX;
                }

                let before_l_idx = (*node_l).prev;
                if before_l_idx != 0xFF {
                    let node_before = nodes_ptr.add(before_l_idx as usize);
                    let id_before = (*node_before).id;
                    let id_l = (*node_l).id;
                    let packed = self.get_pair_packed(id_before, id_l);
                    if packed != u64::MAX {
                        (*node_before).rank = (packed >> 32) as u32;
                        (*node_before).id = packed as u32;
                    } else {
                        (*node_before).rank = u32::MAX;
                    }
                }
            }
        }

        let mut curr_idx = 0usize;
        let mut count = *token_count;
        let slice_len = slice.len();

        while curr_idx != 0xFF {
            if count >= slice_len {
                break;
            }
            unsafe {
                let node = *nodes_ptr.add(curr_idx);
                *slice.get_unchecked_mut(count) = node.id;
                count += 1;
                curr_idx = node.next as usize;
            }
        }
        *token_count = count;
    }
}
