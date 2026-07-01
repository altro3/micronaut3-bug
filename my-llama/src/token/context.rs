use crate::token::bpe_types::BpePair;
use std::collections::BinaryHeap;

pub struct TokenizationContext {
    pub prev: Vec<usize>,
    pub heap: BinaryHeap<BpePair>,
    pub short_prev: [u8; 16],
    pub short_next: [u8; 16],
    pub short_token_ids: [u32; 16],
}

impl TokenizationContext {
    pub fn new() -> Self {
        Self {
            prev: Vec::with_capacity(512),
            heap: BinaryHeap::with_capacity(512),
            short_prev: [0; 16],
            short_next: [0; 16],
            short_token_ids: [0; 16],
        }
    }
}