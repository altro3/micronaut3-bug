use crate::tokenizer::bucket_queue::BucketQueue;

#[repr(align(64))]
pub struct TokenizationContext {
    pub long_ids: Vec<u32>,
    pub long_prev: Vec<i32>,
    pub long_next: Vec<i32>,
    pub heap: BucketQueue,
    pub short_prev: [u8; 16],
    pub short_next: [u8; 16],
    pub short_token_ids: [u32; 16],
}

impl TokenizationContext {
    pub fn new(vocab_size: usize) -> Self {
        let initial_capacity = 2048;

        Self {
            long_ids: Vec::with_capacity(initial_capacity),
            long_prev: Vec::with_capacity(initial_capacity),
            long_next: Vec::with_capacity(initial_capacity),
            heap: BucketQueue::with_capacity(vocab_size, 1024),
            short_prev: [0; 16],
            short_next: [0; 16],
            short_token_ids: [0; 16],
        }
    }
}
