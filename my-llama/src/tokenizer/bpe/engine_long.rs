use super::context::TokenizationContext;
use crate::tokenizer::BpeTokenizer;
use std::hint::unreachable_unchecked;

pub struct LongBpeEngine;

impl LongBpeEngine {
    pub fn merge(data: &BpeTokenizer, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        let nodes_ptr = ctx.nodes.as_mut_ptr();
        let ranks_ptr = ctx.long_ranks.as_mut_ptr();
        let fallback_ptr = data.byte_fallback.as_ptr();

        unsafe {
            for i in 0..len {
                let b = *bytes.get_unchecked(i) as usize;
                let node = nodes_ptr.add(i);
                (*node).id = *fallback_ptr.add(b);
                (*node).next = if i == len - 1 { 0xFFFF } else { (i + 1) as u16 };
                (*node).prev = if i == 0 { 0xFFFF } else { (i - 1) as u16 };
            }

            for i in 0..(len - 1) {
                let packed = data.get_pair_packed((*nodes_ptr.add(i)).id, (*nodes_ptr.add(i + 1)).id);
                *ranks_ptr.add(i) = if packed != u64::MAX { (packed >> 32) as u32 } else { u32::MAX };
            }
            *ranks_ptr.add(len - 1) = u32::MAX;
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = 0xFFFFusize;
            let mut i = 0usize;

            unsafe {
                loop {
                    let r = *ranks_ptr.add(i);
                    if r < min_rank {
                        min_rank = r;
                        best_left = i;
                    }
                    let next_node = (*nodes_ptr.add(i)).next;
                    if next_node == 0xFFFF {
                        break;
                    }
                    i = next_node as usize;
                }
            }

            if min_rank == u32::MAX {
                break;
            }

            unsafe {
                let node_l = nodes_ptr.add(best_left);
                let right_idx = (*node_l).next as usize;
                let node_r = nodes_ptr.add(right_idx);

                let packed = data.get_pair_packed((*node_l).id, (*node_r).id);
                if packed == u64::MAX {
                    unreachable_unchecked();
                }

                (*node_l).id = packed as u32;

                let after_r = (*node_r).next;
                (*node_l).next = after_r;
                if after_r != 0xFFFF {
                    (*nodes_ptr.add(after_r as usize)).prev = best_left as u16;
                }

                *ranks_ptr.add(right_idx) = u32::MAX;

                if after_r != 0xFFFF {
                    let packed_r = data.get_pair_packed((*node_l).id, (*nodes_ptr.add(after_r as usize)).id);
                    *ranks_ptr.add(best_left) = if packed_r != u64::MAX { (packed_r >> 32) as u32 } else { u32::MAX };
                } else {
                    *ranks_ptr.add(best_left) = u32::MAX;
                }

                let prev_idx = (*node_l).prev;
                if prev_idx != 0xFFFF {
                    let p_idx = prev_idx as usize;
                    let packed_l = data.get_pair_packed((*nodes_ptr.add(p_idx)).id, (*node_l).id);
                    *ranks_ptr.add(p_idx) = if packed_l != u64::MAX { (packed_l >> 32) as u32 } else { u32::MAX };
                }
            }
        }

        let mut curr = 0usize;
        let mut count = *token_count;
        let out_tokens_ptr = ctx.tokens_buffer.as_mut_ptr();

        unsafe {
            loop {
                let node = nodes_ptr.add(curr);
                *out_tokens_ptr.add(count) = (*node).id;
                count += 1;
                let next = (*node).next;
                if next == 0xFFFF {
                    break;
                }
                curr = next as usize;
            }
            ctx.tokens_buffer.set_len(count);
        }
        *token_count = count;
    }
}
