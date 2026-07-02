pub struct BucketQueue;

impl BucketQueue {
    #[inline(always)]
    pub fn clear(
        buckets: &mut [u32],
        bitset: &mut [u64],
        next_node: &mut [u32],
        min_rank: usize,
        max_rank_dirty: usize,
        chunk_len: usize,
    ) -> (usize, usize) {
        if min_rank <= max_rank_dirty {
            let start_bucket = min_rank;
            let end_bucket = max_rank_dirty.min(buckets.len() - 1);
            let count = (end_bucket - start_bucket) + 1;

            unsafe {
                std::ptr::write_bytes(buckets.as_mut_ptr().add(start_bucket), 0xFF, count);
            }

            let start_word = start_bucket >> 6;
            let end_word = end_bucket >> 6;
            let word_count = (end_word - start_word) + 1;

            let bitset_ptr = bitset.as_mut_ptr();
            let mut i = 0;
            while i < word_count {
                unsafe {
                    *bitset_ptr.add(start_word + i) = 0;
                }
                i += 1;
            }
        }

        let end_next = chunk_len.min(next_node.len());
        unsafe {
            std::ptr::write_bytes(next_node.as_mut_ptr(), 0xFF, end_next);
        }

        (buckets.len(), 0)
    }

    #[inline(always)]
    pub fn push(
        buckets: &mut [u32],
        bitset: &mut [u64],
        next_node: &mut [u32],
        min_rank: usize,
        max_rank_dirty: usize,
        rank: u32,
        left_idx: usize,
    ) -> (usize, usize) {
        let r = rank as usize;

        let new_min = if r < min_rank { r } else { min_rank };
        let new_max = if r > max_rank_dirty { r } else { max_rank_dirty };

        let word_idx = r >> 6;
        let bit_idx = r & 63;
        let left_idx_u32 = left_idx as u32;

        unsafe {
            let head = *buckets.get_unchecked(r);
            if head == left_idx_u32 {
                return (new_min, new_max);
            }

            *next_node.get_unchecked_mut(left_idx) = head;
            *buckets.get_unchecked_mut(r) = left_idx_u32;
            *bitset.get_unchecked_mut(word_idx) |= 1 << bit_idx;
        }

        (new_min, new_max)
    }

    #[inline(always)]
    pub fn pop_packed(buckets: &mut [u32], bitset: &mut [u64], next_node: &mut [u32], min_rank: usize) -> (u64, usize) {
        let buckets_len = buckets.len();

        if min_rank >= buckets_len {
            return (u64::MAX, min_rank);
        }

        let mut word_idx = min_rank >> 6;
        let bit_offset = min_rank & 63;
        let bitset_len = bitset.len();
        let bitset_ptr = bitset.as_ptr();

        unsafe {
            let mut word = *bitset_ptr.add(word_idx) & (!0u64 << bit_offset);

            if word == 0 {
                word_idx += 1;

                while word_idx + 3 < bitset_len {
                    let w0 = *bitset_ptr.add(word_idx);
                    let w1 = *bitset_ptr.add(word_idx + 1);
                    let w2 = *bitset_ptr.add(word_idx + 2);
                    let w3 = *bitset_ptr.add(word_idx + 3);

                    if (w0 | w1 | w2 | w3) != 0 {
                        if w0 != 0 {
                            word = w0;
                            break;
                        }
                        if w1 != 0 {
                            word = w1;
                            word_idx += 1;
                            break;
                        }
                        if w2 != 0 {
                            word = w2;
                            word_idx += 2;
                            break;
                        }
                        word = w3;
                        word_idx += 3;
                        break;
                    }
                    word_idx += 4;
                }

                if word == 0 {
                    while word_idx < bitset_len {
                        word = *bitset_ptr.add(word_idx);
                        if word != 0 {
                            break;
                        }
                        word_idx += 1;
                    }

                    if word == 0 {
                        return (u64::MAX, buckets_len);
                    }
                }
            }

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

            (((actual_rank as u64) << 32) | (head as u64), actual_rank)
        }
    }
}
