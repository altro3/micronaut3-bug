use crate::token::bpe_types::BpePair;

pub struct BucketQueue {
    buckets: Vec<u32>,
    next_node: Vec<u32>,
    min_rank: usize,
}

impl BucketQueue {
    pub fn with_capacity(vocab_size: usize, chunk_len: usize) -> Self {
        Self {
            buckets: vec![u32::MAX; vocab_size],
            next_node: vec![u32::MAX; chunk_len.max(512)],
            min_rank: vocab_size,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.min_rank = self.buckets.len();
        self.next_node.fill(u32::MAX);
    }

    #[inline(always)]
    pub fn next_node_len(&self) -> usize {
        self.next_node.len()
    }

    #[inline(always)]
    pub fn reserve_chunk_len(&mut self, new_len: usize) {
        if new_len > self.next_node.len() {
            self.next_node.resize(new_len, u32::MAX);
        }
    }

    #[inline(always)]
    pub fn push(&mut self, rank: u32, left_idx: usize) {
        let r = rank as usize;
        if r < self.min_rank {
            self.min_rank = r;
        }

        if left_idx >= self.next_node.len() {
            self.next_node.resize(left_idx + 256, u32::MAX);
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
    pub fn pop(&mut self) -> Option<BpePair> {
        let len = self.buckets.len();
        while self.min_rank < len {
            let head = unsafe { *self.buckets.get_unchecked(self.min_rank) };
            if head != u32::MAX {
                unsafe {
                    let next = *self.next_node.get_unchecked(head as usize);
                    if next == head || next == u32::MAX {
                        *self.buckets.get_unchecked_mut(self.min_rank) = u32::MAX;
                    } else {
                        *self.buckets.get_unchecked_mut(self.min_rank) = next;
                    }

                    *self.next_node.get_unchecked_mut(head as usize) = u32::MAX;

                    return Some(BpePair {
                        rank: self.min_rank as u32,
                        left_idx: head as usize,
                    });
                }
            }
            self.min_rank += 1;
        }
        None
    }
}
