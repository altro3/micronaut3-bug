use std::arch::x86_64::*;

#[repr(align(64))]
pub struct BucketQueue {
    pub buckets: Vec<u32>,
    pub bitset: Vec<u64>,
    pub next_node: Vec<u32>,
    pub min_rank_dirty: usize,
    pub max_rank_dirty: usize,
}

impl BucketQueue {
    #[inline(always)]
    pub fn with_capacity(vocab_size: usize, max_chunk_capacity: usize) -> Self {
        let bitset_len = (vocab_size + 63) >> 6;
        Self {
            buckets: vec![u32::MAX; vocab_size],
            bitset: vec![0u64; bitset_len],
            next_node: vec![u32::MAX; max_chunk_capacity],
            min_rank_dirty: vocab_size,
            max_rank_dirty: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self, chunk_len: usize) {
        if self.min_rank_dirty <= self.max_rank_dirty {
            let start_bucket = self.min_rank_dirty;
            let end_bucket = self.max_rank_dirty.min(self.buckets.len() - 1);

            if start_bucket <= end_bucket {
                self.buckets[start_bucket..=end_bucket].fill(u32::MAX);

                let start_word = start_bucket >> 6;
                let end_word = (end_bucket >> 6).min(self.bitset.len() - 1);
                if start_word <= end_word {
                    self.bitset[start_word..=end_word].fill(0);
                }
            }
        }

        let end_next = chunk_len.min(self.next_node.len());
        self.next_node[..end_next].fill(u32::MAX);

        self.min_rank_dirty = self.buckets.len();
        self.max_rank_dirty = 0;
    }

    #[inline(always)]
    pub fn push(&mut self, rank: u32, left_idx: usize) {
        let r = rank as usize;

        self.min_rank_dirty = std::cmp::min(self.min_rank_dirty, r);
        self.max_rank_dirty = std::cmp::max(self.max_rank_dirty, r);

        let word_idx = r >> 6;
        let bit_idx = r & 63;

        unsafe {
            let head_ptr = self.buckets.get_unchecked_mut(r);
            let head = *head_ptr;

            *self.next_node.get_unchecked_mut(left_idx) = head;
            *head_ptr = left_idx as u32;
            *self.bitset.get_unchecked_mut(word_idx) |= 1u64 << bit_idx;
        }
    }

    #[inline(always)]
    pub fn pop_packed(&mut self) -> u64 {
        let buckets_len = self.buckets.len();
        if self.min_rank_dirty >= buckets_len {
            return u64::MAX;
        }

        let mut word_idx = self.min_rank_dirty >> 6;
        let bit_offset = self.min_rank_dirty & 63;
        let bitset_len = self.bitset.len();

        unsafe {
            let bitset_ptr = self.bitset.as_ptr();
            let raw_word = *bitset_ptr.add(word_idx);
            let mut word = raw_word & (!0u64 << bit_offset);

            if word == 0 {
                word_idx += 1;

                while word_idx + 4 <= bitset_len {
                    let current_ptr = bitset_ptr.add(word_idx);
                    let vec_data = std::ptr::read_unaligned(current_ptr as *const __m256i);
                    let zeroes = _mm256_setzero_si256();
                    let cmp = _mm256_cmpeq_epi64(vec_data, zeroes);
                    let mask = _mm256_movemask_epi8(cmp) as u32;

                    if mask != 0xFFFFFFFF {
                        let inv_mask = !mask;
                        let first_nonzero_byte_idx = inv_mask.trailing_zeros() as usize;
                        let target_word_offset = first_nonzero_byte_idx >> 3;

                        word_idx += target_word_offset;
                        word = *bitset_ptr.add(word_idx);
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
                        self.min_rank_dirty = buckets_len;
                        return u64::MAX;
                    }
                }
            }

            let tz = word.trailing_zeros() as usize;
            let actual_rank = (word_idx << 6) + tz;

            if actual_rank >= buckets_len {
                self.min_rank_dirty = buckets_len;
                return u64::MAX;
            }

            let bucket_ptr = self.buckets.get_unchecked_mut(actual_rank);
            let head = *bucket_ptr;
            let head_idx = head as usize;

            let next_ptr = self.next_node.get_unchecked_mut(head_idx);
            let next = *next_ptr;

            let is_empty_mask = ((next == u32::MAX) as u64).wrapping_neg();
            let bit_clear_mask = !((1u64 << tz) & is_empty_mask);

            *self.bitset.get_unchecked_mut(word_idx) &= bit_clear_mask;
            *bucket_ptr = next;
            *next_ptr = u32::MAX;

            if next == u32::MAX {
                self.min_rank_dirty = actual_rank + 1;
            } else {
                self.min_rank_dirty = actual_rank;
            }

            ((actual_rank as u64) << 32) | (head as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_queue_isolated() {
        let vocab_size = 1000;
        let max_chunk_capacity = 100;
        let mut queue = BucketQueue::with_capacity(vocab_size, max_chunk_capacity);

        // Пушим минимальный набор
        queue.push(50, 5);
        queue.push(100, 10);

        let p1 = queue.pop_packed();
        assert_ne!(p1, u64::MAX);
        assert_eq!((p1 >> 32) as u32, 50);
        assert_eq!((p1 & 0xFFFFFFFF) as usize, 5);

        let p2 = queue.pop_packed();
        assert_ne!(p2, u64::MAX);
        assert_eq!((p2 >> 32) as u32, 100);
        assert_eq!((p2 & 0xFFFFFFFF) as usize, 10);

        assert_eq!(queue.pop_packed(), u64::MAX);
    }
}
