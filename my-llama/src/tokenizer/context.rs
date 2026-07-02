use crate::tokenizer::bucket_queue::BucketQueue;
use crate::tokenizer::simd_splitter::TokenSpan;

#[repr(align(64))]
pub struct TokenizationContext {
    pub tokens_buffer: Vec<u32>,
    pub spans_buffer: Vec<TokenSpan>,
    pub long_ids: Vec<u32>,
    pub long_prev: Vec<i32>,
    pub long_next: Vec<i32>,
    pub heap: BucketQueue,
    pub short_prev: [u8; 16],
    pub short_next: [u8; 16],
    pub short_token_ids: [u32; 16],
}

impl TokenizationContext {
    pub fn new(vocab_size: usize, max_chunk_capacity: usize) -> Self {
        Self {
            tokens_buffer: Vec::with_capacity(max_chunk_capacity * 2),
            spans_buffer: vec![TokenSpan { start: 0, end: 0 }; max_chunk_capacity],
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
        if text_len + 1 > self.tokens_buffer.capacity() {
            self.tokens_buffer.reserve((text_len + 1) - self.tokens_buffer.len());
        }
        unsafe {
            self.tokens_buffer.set_len(text_len + 1);
        }

        let required_spans = (text_len / 2).max(16);
        if required_spans > self.spans_buffer.len() {
            self.spans_buffer.resize(required_spans, TokenSpan { start: 0, end: 0 });
        }
    }
}
