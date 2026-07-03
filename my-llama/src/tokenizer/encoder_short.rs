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
    pub fn encode_single_chunk(&self, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        if len == 1 {
            unsafe {
                let out_tokens_ptr = ctx.tokens_buffer.as_mut_ptr();
                *out_tokens_ptr.add(*token_count) = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(0) as usize);
                *token_count += 1;
            }
            return;
        }

        if len <= 16 {
            self.encode_short_chunk(bytes, token_count, ctx);
        } else {
            self.encode_long_chunk(bytes, token_count, ctx);
        }
    }

    fn encode_short_chunk(&self, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
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

        loop {
            unsafe {
                let mut curr = 0usize;
                loop {
                    let node_curr = nodes_ptr.add(curr);
                    let r_idx = (*node_curr).next;

                    if r_idx != 0xFF {
                        let node_next = nodes_ptr.add(r_idx as usize);
                        let packed = self.get_pair_packed((*node_curr).id, (*node_next).id);
                        (*node_curr).rank = if packed != u64::MAX { (packed >> 32) as u32 } else { u32::MAX };
                        curr = r_idx as usize;
                    } else {
                        (*node_curr).rank = u32::MAX;
                        break;
                    }
                }
            }

            let mut min_rank: u32 = u32::MAX;
            let mut best_left: usize = 0xFF;

            unsafe {
                let mut curr = 0usize;
                loop {
                    let r = (*nodes_ptr.add(curr)).rank;
                    if r < min_rank {
                        min_rank = r;
                        best_left = curr;
                    }
                    let next_node = (*nodes_ptr.add(curr)).next;
                    if next_node == 0xFF {
                        break;
                    }
                    curr = next_node as usize;
                }
            }

            if min_rank == u32::MAX || best_left == 0xFF {
                break;
            }

            unsafe {
                let node_l = nodes_ptr.add(best_left);
                let r_idx = (*node_l).next as usize;
                let node_r = nodes_ptr.add(r_idx);

                let packed_merge = self.get_pair_packed((*node_l).id, (*node_r).id);
                (*node_l).id = packed_merge as u32;

                let after_r_idx = (*node_r).next;
                (*node_l).next = after_r_idx;

                if after_r_idx != 0xFF {
                    (*nodes_ptr.add(after_r_idx as usize)).prev = best_left as u8;
                }

                (*node_r).rank = u32::MAX;
                (*node_r).next = 0xFF;
                (*node_r).prev = 0xFF;
            }
        }

        let mut curr_idx = 0usize;
        let mut count = *token_count;
        let out_tokens_ptr = ctx.tokens_buffer.as_mut_ptr();

        unsafe {
            while curr_idx != 0xFF {
                let node = nodes_ptr.add(curr_idx);
                *out_tokens_ptr.add(count) = (*node).id;
                count += 1;
                curr_idx = (*node).next as usize;
            }
            ctx.tokens_buffer.set_len(count);
        }
        *token_count = count;
    }
}
