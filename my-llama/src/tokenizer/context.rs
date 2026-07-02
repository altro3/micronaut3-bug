use crate::tokenizer::bucket_queue::BucketQueue;

pub struct TokenizationContext {
    pub prev: Vec<usize>,
    pub heap: BucketQueue,
    pub short_prev: [u8; 16],
    pub short_next: [u8; 16],
    pub short_token_ids: [u32; 16],
}

impl TokenizationContext {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            prev: Vec::with_capacity(4096),
            heap: BucketQueue::with_capacity(vocab_size, 1024),
            short_prev: [0; 16],
            short_next: [0; 16],
            short_token_ids: [0; 16],
        }
    }
}
