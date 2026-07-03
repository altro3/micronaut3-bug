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
                }
            }
        }

        loop {
            let mut min_rank: u32 = u32::MAX;
            let mut best_left: usize = 0xFF;

            unsafe {
                macro_rules! check_rank {
                    ($idx:expr) => {
                        if $idx < len {
                            let r = (*nodes_ptr.add($idx)).rank;
                            if r < min_rank {
                                min_rank = r;
                                best_left = $idx;
                            }
                        }
                    };
                }

                check_rank!(0);
                check_rank!(1);
                check_rank!(2);
                check_rank!(3);
                check_rank!(4);
                check_rank!(5);
                check_rank!(6);
                check_rank!(7);
                check_rank!(8);
                check_rank!(9);
                check_rank!(10);
                check_rank!(11);
                check_rank!(12);
                check_rank!(13);
                check_rank!(14);
                check_rank!(15);
            }

            if min_rank == u32::MAX || best_left == 0xFF {
                break;
            }

            unsafe {
                let node_l = nodes_ptr.add(best_left);
                let r_idx = (*node_l).next as usize;
                let node_r = nodes_ptr.add(r_idx);

                let packed_merge = self.get_pair_packed((*node_l).id, (*node_r).id);
                let new_token_id = packed_merge as u32;
                (*node_l).id = new_token_id;

                let after_r_idx = (*node_r).next;
                (*node_l).next = after_r_idx;

                if after_r_idx != 0xFF {
                    (*nodes_ptr.add(after_r_idx as usize)).prev = best_left as u8;
                }

                (*node_r).rank = u32::MAX;
                (*node_r).next = 0xFF;
                (*node_r).prev = 0xFF;

                if after_r_idx != 0xFF {
                    let id_after = (*nodes_ptr.add(after_r_idx as usize)).id;
                    let packed = self.get_pair_packed(new_token_id, id_after);
                    (*node_l).rank = if packed != u64::MAX { (packed >> 32) as u32 } else { u32::MAX };
                } else {
                    (*node_l).rank = u32::MAX;
                }

                let before_l_idx = (*node_l).prev;
                if before_l_idx != 0xFF {
                    let node_before = nodes_ptr.add(before_l_idx as usize);
                    let packed = self.get_pair_packed((*node_before).id, new_token_id);
                    (*node_before).rank = if packed != u64::MAX { (packed >> 32) as u32 } else { u32::MAX };
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
                let node = nodes_ptr.add(curr_idx);
                *slice.get_unchecked_mut(count) = (*node).id;
                count += 1;
                curr_idx = (*node).next as usize;
            }
        }
        *token_count = count;
    }
}
