use super::context::TokenizationContext;
use crate::tokenizer::BpeTokenizer;

pub struct ShortBpeEngine;

impl ShortBpeEngine {
    #[inline(always)]
    pub fn merge(data: &BpeTokenizer, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        unsafe {
            let out_ptr = ctx.tokens_buffer.as_mut_ptr();
            let mut count = *token_count;

            let trie_ptr = data.trie_nodes.as_ptr();
            let root_table_ptr = data.trie_root_offsets.as_ptr();
            let fallback_ptr = data.byte_fallback.as_ptr();

            let mut start_idx = 0usize;

            while start_idx < len {
                let mut best_token_id = u32::MAX;
                let mut best_len = 0usize;

                let first_byte = *bytes.get_unchecked(start_idx) as usize;
                let curr_offset = *root_table_ptr.add(first_byte);

                if curr_offset != u32::MAX {
                    let mut curr_idx = start_idx + 1;
                    let first_node = *trie_ptr.add(curr_offset as usize);

                    if first_node.token_id != u32::MAX {
                        best_token_id = first_node.token_id;
                        best_len = 1;
                    }

                    let mut next_children = first_node.children_offset;

                    while curr_idx < len && next_children != u32::MAX {
                        let next_byte = *bytes.get_unchecked(curr_idx) as usize;
                        let child_node_ptr = trie_ptr.add((next_children as usize) + next_byte);
                        let child_node = *child_node_ptr;

                        if child_node.token_id != u32::MAX {
                            best_token_id = child_node.token_id;
                            best_len = curr_idx - start_idx + 1;
                        }

                        next_children = child_node.children_offset;
                        curr_idx += 1;
                    }
                }

                if best_token_id != u32::MAX {
                    *out_ptr.add(count) = best_token_id;
                    count += 1;
                    start_idx += best_len;
                } else {
                    let b = *bytes.get_unchecked(start_idx) as usize;
                    *out_ptr.add(count) = *fallback_ptr.add(b);
                    count += 1;
                    start_idx += 1;
                }
            }

            *token_count = count;
            ctx.tokens_buffer.set_len(count);
        }
    }
}
