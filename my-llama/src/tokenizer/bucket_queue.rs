pub struct BucketQueue {
    buckets: Vec<u32>,
    next_node: Vec<u32>,
    min_rank: usize,
    max_rank_dirty: usize,
}

impl BucketQueue {
    pub fn with_capacity(vocab_size: usize, chunk_len: usize) -> Self {
        Self {
            buckets: vec![u32::MAX; vocab_size],
            next_node: vec![u32::MAX; chunk_len.max(512)],
            min_rank: vocab_size,
            max_rank_dirty: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self, chunk_len: usize) {
        if self.min_rank <= self.max_rank_dirty {
            let end = self.max_rank_dirty.min(self.buckets.len() - 1);
            unsafe {
                let ptr = self.buckets.as_mut_ptr().add(self.min_rank);
                let count = (end - self.min_rank) + 1;
                std::ptr::write_bytes(ptr, 0xFF, count);
            }
        }

        let end_next = chunk_len.min(self.next_node.len());
        unsafe {
            std::ptr::write_bytes(self.next_node.as_mut_ptr(), 0xFF, end_next);
        }

        self.min_rank = self.buckets.len();
        self.max_rank_dirty = 0;
    }

    #[inline(always)]
    pub fn next_node_len(&self) -> usize {
        self.next_node.len()
    }

    #[inline(always)]
    pub fn reserve_chunk_len(&mut self, new_len: usize) {
        if new_len > self.next_node.len() {
            let additional = new_len - self.next_node.len();
            self.next_node.reserve(additional);
            unsafe {
                let old_len = self.next_node.len();
                self.next_node.set_len(new_len);
                let ptr = self.next_node.as_mut_ptr().add(old_len);
                // Здесь вы изначально написали правильно — additional передается без умножения
                std::ptr::write_bytes(ptr, 0xFF, additional);
            }
        }
    }

    #[inline(always)]
    pub fn push(&mut self, rank: u32, left_idx: usize) {
        let r = rank as usize;
        if r < self.min_rank {
            self.min_rank = r;
        }
        if r > self.max_rank_dirty {
            self.max_rank_dirty = r;
        }

        unsafe {
            let head = *self.buckets.get_unchecked(r);
            if head == left_idx as u32 {
                return;
            }

            *self.next_node.get_unchecked_mut(left_idx) = head;
            *self.buckets.get_unchecked_mut(r) = left_idx as u32;
        }
    }

    #[inline(always)]
    pub fn pop_packed(&mut self) -> u64 {
        let len = self.buckets.len();

        while self.min_rank < len {
            let head = unsafe { *self.buckets.get_unchecked(self.min_rank) };

            if head != u32::MAX {
                unsafe {
                    let head_idx = head as usize;
                    let next = *self.next_node.get_unchecked(head_idx);

                    *self.buckets.get_unchecked_mut(self.min_rank) = next;
                    *self.next_node.get_unchecked_mut(head_idx) = u32::MAX;

                    return ((self.min_rank as u64) << 32) | (head as u64);
                }
            }
            self.min_rank += 1;
        }

        u64::MAX
    }
}
