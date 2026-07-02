use crate::tokenizer::bucket_queue::BucketQueue;

#[repr(align(64))]
pub struct TokenizationContext {
    pub tokens_buffer: Vec<u32>,
    pub long_ids: Vec<u32>,
    pub long_prev: Vec<i32>,
    pub long_next: Vec<i32>,
    pub heap: BucketQueue,
    pub short_token_ids: [u32; 16],
    pub short_prev: [u8; 16],
    pub short_next: [u8; 16],
}

impl TokenizationContext {
    pub fn new(vocab_size: usize, max_chunk_capacity: usize) -> Self {
        Self {
            tokens_buffer: Vec::with_capacity(max_chunk_capacity * 2),
            long_ids: Vec::with_capacity(max_chunk_capacity),
            long_prev: Vec::with_capacity(max_chunk_capacity),
            long_next: Vec::with_capacity(max_chunk_capacity),
            heap: BucketQueue::with_capacity(vocab_size, max_chunk_capacity),
            short_prev: [0; 16],
            short_next: [0; 16],
            short_token_ids: [0; 16],
        }
    }

    #[inline(always)]
    pub fn reset(&mut self, text_len: usize) {
        let required_len = text_len + 1;
        let current_capacity = self.tokens_buffer.capacity();

        if required_len <= current_capacity {
            unsafe {
                self.tokens_buffer.set_len(required_len);
            }
        } else {
            let new_capacity = required_len.next_power_of_two().max(current_capacity * 2);
            self.tokens_buffer.reserve_exact(new_capacity - self.tokens_buffer.len());
            unsafe {
                self.tokens_buffer.set_len(required_len);
            }
        }

        unsafe {
            self.long_ids.set_len(0);
            self.long_prev.set_len(0);
            self.long_next.set_len(0);
        }
    }
}
