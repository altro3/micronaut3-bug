pub struct BucketQueue;

impl BucketQueue {
    #[inline(always)]
    pub fn clear(buckets: &mut [u32], bitset: &mut [u64], next_node: &mut [u32], min_rank: &mut usize, max_rank_dirty: &mut usize, chunk_len: usize) {
        if *min_rank <= *max_rank_dirty {
            let start_bucket = *min_rank;
            let end_bucket = (*max_rank_dirty).min(buckets.len() - 1);
            let count = (end_bucket - start_bucket) + 1;

            unsafe {
                std::ptr::write_bytes(buckets.as_mut_ptr().add(start_bucket), 0xFF, count);
            }

            let start_word = start_bucket >> 6;
            let end_word = end_bucket >> 6;
            let word_count = (end_word - start_word) + 1;
            unsafe {
                std::ptr::write_bytes(bitset.as_mut_ptr().add(start_word), 0x00, word_count * 8);
            }
        }

        let end_next = chunk_len.min(next_node.len());
        unsafe {
            std::ptr::write_bytes(next_node.as_mut_ptr(), 0xFF, end_next);
        }

        *min_rank = buckets.len();
        *max_rank_dirty = 0;
    }

    #[inline(always)]
    pub fn push(
        buckets: &mut [u32],
        bitset: &mut [u64],
        next_node: &mut [u32],
        min_rank: &mut usize,
        max_rank_dirty: &mut usize,
        rank: u32,
        left_idx: usize,
    ) {
        let r = rank as usize;
        if r < *min_rank {
            *min_rank = r;
        }
        if r > *max_rank_dirty {
            *max_rank_dirty = r;
        }

        unsafe {
            let head = *buckets.get_unchecked(r);
            if head == left_idx as u32 {
                return;
            }

            *next_node.get_unchecked_mut(left_idx) = head;
            *buckets.get_unchecked_mut(r) = left_idx as u32;

            let word_idx = r >> 6;
            let bit_idx = r & 63;
            *bitset.get_unchecked_mut(word_idx) |= 1 << bit_idx;
        }
    }

    #[inline(always)]
    pub fn pop_packed(buckets: &mut [u32], bitset: &mut [u64], next_node: &mut [u32], min_rank: &mut usize) -> u64 {
        let buckets_len = buckets.len();
        let mut curr_rank = *min_rank;

        while curr_rank < buckets_len {
            let word_idx = curr_rank >> 6;
            let bit_offset = curr_rank & 63;

            unsafe {
                let word = *bitset.get_unchecked(word_idx) & (!0u64 << bit_offset);

                if word != 0 {
                    let tz = word.trailing_zeros() as usize;
                    let actual_rank = (word_idx << 6) + tz;

                    let head = *buckets.get_unchecked(actual_rank);
                    let head_idx = head as usize;
                    let next = *next_node.get_unchecked(head_idx);

                    *buckets.get_unchecked_mut(actual_rank) = next;
                    *next_node.get_unchecked_mut(head_idx) = u32::MAX;

                    if next == u32::MAX {
                        *bitset.get_unchecked_mut(word_idx) &= !(1 << tz);
                    }

                    *min_rank = actual_rank;
                    return ((actual_rank as u64) << 32) | (head as u64);
                }
            }

            curr_rank = (word_idx + 1) << 6;
        }

        *min_rank = buckets_len;
        u64::MAX
    }
}
