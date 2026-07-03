use crate::tokenizer::bucket_queue::BucketQueue;

#[repr(align(64))]
pub struct TokenizationContext {
    pub short_token_ids: [u32; 16],
    pub short_prev: [u8; 16],
    pub short_next: [u8; 16],

    pub tokens_buffer: Vec<u32>,
    pub tokens_lens_buffer: Vec<u32>,
    pub long_ids: Vec<u32>,
    pub long_prev: Vec<i32>,
    pub long_next: Vec<i32>,

    pub heap: BucketQueue,
}

impl TokenizationContext {
    pub fn new(vocab_size: usize, max_chunk_capacity: usize) -> Self {
        Self {
            short_token_ids: [0; 16],
            short_prev: [0; 16],
            short_next: [0; 16],
            tokens_buffer: Vec::with_capacity(max_chunk_capacity * 2),
            tokens_lens_buffer: Vec::with_capacity(max_chunk_capacity * 2),
            long_ids: Vec::with_capacity(max_chunk_capacity),
            long_prev: Vec::with_capacity(max_chunk_capacity),
            long_next: Vec::with_capacity(max_chunk_capacity),
            heap: BucketQueue::with_capacity(vocab_size, max_chunk_capacity),
        }
    }

    #[inline(always)]
    pub fn reset(&mut self, text_len: usize) {
        let required_len = text_len + 1;

        if required_len > self.tokens_buffer.capacity() {
            let new_capacity = required_len.next_power_of_two();
            self.tokens_buffer.reserve(new_capacity - self.tokens_buffer.len());
            self.tokens_lens_buffer.reserve(new_capacity - self.tokens_lens_buffer.len());
        }

        self.tokens_buffer.resize(required_len, 0);
        self.tokens_lens_buffer.resize(required_len, 0);

        if text_len > self.long_ids.capacity() {
            let new_long_cap = text_len.next_power_of_two();
            self.long_ids.reserve(new_long_cap - self.long_ids.len());
            self.long_prev.reserve(new_long_cap - self.long_prev.len());
            self.long_next.reserve(new_long_cap - self.long_next.len());
        }

        self.long_ids.resize(text_len, 0);
        self.long_prev.resize(text_len, -1);
        self.long_next.resize(text_len, -1);

        self.heap.clear(text_len);
    }
}
