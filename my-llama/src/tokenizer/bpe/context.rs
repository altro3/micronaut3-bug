use super::bpe_types::FlatBpeNode;
use crate::tokenizer::bpe::heap_types::MergePair;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct BpeCacheEntry {
    pub key: u64,
    pub val: u64,
}

#[repr(C, align(64))]
pub struct TokenizationContext {
    pub short_token_ids: [u32; 16],
    pub short_prev: [u8; 16],
    pub short_next: [u8; 16],

    pub bpe_direct_cache: Vec<BpeCacheEntry>,

    pub nodes: Vec<FlatBpeNode>,
    pub long_ranks: Vec<u32>,
    pub tokens_buffer: Vec<u32>,

    pub chunk_offsets: Vec<u32>,
    pub tokens_lens_buffer: Vec<u32>,

    pub vocab_size: usize,
    pub max_capacity: usize,
    pub bpe_heap: Vec<MergePair>,
    pub bpe_generations: Vec<u16>,
}

impl TokenizationContext {
    pub fn new(vocab_size: usize, max_chunk_capacity: usize) -> Self {
        Self {
            short_token_ids: [0; 16],
            short_prev: [0; 16],
            short_next: [0; 16],

            // Инициализируем 16к ячеек интерливинг-кэша.
            // Ключ u64::MAX гарантирует отсутствие ложных попаданий на старте.
            bpe_direct_cache: vec![
                BpeCacheEntry {
                    key: u64::MAX,
                    val: u64::MAX
                };
                16384
            ],

            chunk_offsets: Vec::with_capacity(max_chunk_capacity),
            tokens_lens_buffer: Vec::with_capacity(max_chunk_capacity),
            tokens_buffer: Vec::with_capacity(max_chunk_capacity * 2),
            long_ranks: vec![u32::MAX; max_chunk_capacity + 64],
            nodes: vec![
                FlatBpeNode {
                    id: 0,
                    next: 0xFFFF,
                    prev: 0xFFFF
                };
                max_chunk_capacity + 64
            ],
            vocab_size,
            max_capacity: max_chunk_capacity,
            bpe_heap: Vec::with_capacity(max_chunk_capacity),
            bpe_generations: Vec::with_capacity(max_chunk_capacity),
        }
    }

    #[inline(always)]
    pub fn reset_all(&mut self, text_len: usize) {
        self.tokens_buffer.clear();

        let required_len = text_len + 32;
        let safe_len = required_len.min(self.max_capacity + 64);

        if safe_len > self.nodes.len() {
            self.nodes.resize(
                safe_len,
                FlatBpeNode {
                    id: 0,
                    next: 0xFFFF,
                    prev: 0xFFFF,
                },
            );
            self.long_ranks.resize(safe_len, u32::MAX);
        }

        if safe_len > self.chunk_offsets.capacity() {
            let new_cap = safe_len.next_power_of_two();
            self.chunk_offsets.reserve(new_cap - self.chunk_offsets.len());
            self.tokens_lens_buffer.reserve(new_cap - self.tokens_lens_buffer.len());
        }

        unsafe {
            self.chunk_offsets.set_len(safe_len);
            self.tokens_lens_buffer.set_len(safe_len);
        }
    }

    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }
}

unsafe impl Send for TokenizationContext {}
unsafe impl Sync for TokenizationContext {}
