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
        Self {
            buckets: vec![u32::MAX; vocab_size],
            bitset: vec![0u64; (vocab_size + 63) >> 6],
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
            let count = (end_bucket - start_bucket) + 1;

            unsafe {
                std::ptr::write_bytes(self.buckets.as_mut_ptr().add(start_bucket), 0xFF, count);
            }

            let start_word = start_bucket >> 6;
            let end_word = end_bucket >> 6;

            let word_count = (end_word - start_word) + 1;
            unsafe {
                std::ptr::write_bytes(self.bitset.as_mut_ptr().add(start_word), 0, word_count * 8);
            }
        }

        let end_next = chunk_len.min(self.next_node.len());
        unsafe {
            std::ptr::write_bytes(self.next_node.as_mut_ptr(), 0xFF, end_next);
        }

        self.min_rank_dirty = self.buckets.len();
        self.max_rank_dirty = 0;
    }

    #[inline(always)]
    pub fn push(&mut self, rank: u32, left_idx: usize) {
        let r = rank as usize;

        if r < self.min_rank_dirty {
            self.min_rank_dirty = r;
        }
        if r > self.max_rank_dirty {
            self.max_rank_dirty = r;
        }

        let word_idx = r >> 6;
        let bit_idx = r & 63;

        unsafe {
            let head = *self.buckets.get_unchecked(r);
            *self.next_node.get_unchecked_mut(left_idx) = head;
            *self.buckets.get_unchecked_mut(r) = left_idx as u32;
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
        let bitset_ptr = self.bitset.as_ptr();

        unsafe {
            let raw_word = *bitset_ptr.add(word_idx);
            let mut word = raw_word & (!0u64 << bit_offset);

            if word == 0 {
                word_idx += 1;

                while word_idx + 4 <= bitset_len {
                    let vec_data = _mm256_loadu_si256(bitset_ptr.add(word_idx) as *const __m256i);
                    let zeroes = _mm256_setzero_si256();
                    let cmp = _mm256_cmpeq_epi64(vec_data, zeroes);
                    let mask = _mm256_movemask_epi8(cmp) as u32;

                    if mask != 0xFFFFFFFF {
                        for offset in 0..4 {
                            let w = *bitset_ptr.add(word_idx + offset);
                            if w != 0 {
                                word_idx += offset;
                                word = w;
                                break;
                            }
                        }
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
            self.min_rank_dirty = actual_rank;

            let head = *self.buckets.get_unchecked(actual_rank);
            let head_idx = head as usize;
            let next = *self.next_node.get_unchecked(head_idx);

            let origin_word = *self.bitset.get_unchecked(word_idx);
            let mask_update = !(1u64 << tz);
            let updated_bitset_word = origin_word & mask_update;

            let final_bitset_word = if next == u32::MAX { updated_bitset_word } else { origin_word };

            *self.bitset.get_unchecked_mut(word_idx) = final_bitset_word;
            *self.buckets.get_unchecked_mut(actual_rank) = next;
            *self.next_node.get_unchecked_mut(head_idx) = u32::MAX;

            ((actual_rank as u64) << 32) | (head as u64)
        }
    }
}
