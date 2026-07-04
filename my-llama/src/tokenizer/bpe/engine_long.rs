use super::context::TokenizationContext;
use super::heap_types::MergePair;
use crate::tokenizer::BpeTokenizer;
use std::collections::BinaryHeap;

pub struct LongBpeEngine;

impl LongBpeEngine {
    #[inline(always)]
    pub fn merge(data: &BpeTokenizer, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        let nodes_ptr = ctx.nodes.as_mut_ptr();
        let fallback_ptr = data.byte_fallback.as_ptr();

        let mut heap = BinaryHeap::with_capacity(len);
        let mut generations = vec![0u16; len];

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
                if packed != u64::MAX {
                    heap.push(MergePair { rank: (packed >> 32) as u32, left_idx: i as u16, generation: 0 });
                }
            }

            while let Some(pair) = heap.pop() {
                let left_idx = pair.left_idx as usize;
                if pair.generation != *generations.get_unchecked(left_idx) { continue; }

                let node_l = nodes_ptr.add(left_idx);
                let right_idx = (*node_l).next as usize;
                if right_idx == 0xFFFF { continue; }
                let node_r = nodes_ptr.add(right_idx);

                let packed = data.get_pair_packed((*node_l).id, (*node_r).id);
                if packed == u64::MAX || (packed >> 32) as u32 != pair.rank { continue; }

                (*node_l).id = packed as u32;
                let after_r = (*node_r).next;
                (*node_l).next = after_r;

                *generations.get_unchecked_mut(left_idx) += 1;
                *generations.get_unchecked_mut(right_idx) += 1;
                let current_gen = *generations.get_unchecked(left_idx);

                if after_r != 0xFFFF {
                    let after_r_idx = after_r as usize;
                    (*nodes_ptr.add(after_r_idx)).prev = left_idx as u16;

                    let packed_r = data.get_pair_packed((*node_l).id, (*nodes_ptr.add(after_r_idx)).id);
                    if packed_r != u64::MAX {
                        heap.push(MergePair { rank: (packed_r >> 32) as u32, left_idx: left_idx as u16, generation: current_gen });
                    }
                }

                let prev_idx = (*node_l).prev;
                if prev_idx != 0xFFFF {
                    let p_idx = prev_idx as usize;
                    *generations.get_unchecked_mut(p_idx) += 1;

                    let packed_l = data.get_pair_packed((*nodes_ptr.add(p_idx)).id, (*node_l).id);
                    if packed_l != u64::MAX {
                        heap.push(MergePair { rank: (packed_l >> 32) as u32, left_idx: prev_idx, generation: *generations.get_unchecked(p_idx) });
                    }
                }
            }

            let mut curr = 0usize;
            let mut count = *token_count;
            let out_ptr = ctx.tokens_buffer.as_mut_ptr();
            loop {
                let node = nodes_ptr.add(curr);
                *out_ptr.add(count) = (*node).id;
                count += 1;
                let next = (*node).next;
                if next == 0xFFFF { break; }
                curr = next as usize;
            }
            *token_count = count;
            ctx.tokens_buffer.set_len(count);
        }
    }
}
