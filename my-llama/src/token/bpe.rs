use super::bpe_types::BpeValue;
use super::context::TokenizationContext;
use super::simd_splitter::SimdSplitter;
use crate::cuda::PinnedHostBuffer;
use rustc_hash::FxHashMap;

pub struct BpeTokenizer {
    pair_ranks: FxHashMap<u64, BpeValue>,
    byte_pair_ranks: Box<[[u64; 256]; 256]>,
    byte_fallback: [u32; 256],
    id_to_byte: [i16; 512],
    pub eos_token_id: u32,
    vocab_size: usize,
}

impl BpeTokenizer {
    pub fn new(pair_ranks: FxHashMap<u64, BpeValue>, byte_fallback: [u32; 256], eos_token_id: u32) -> Self {
        let mut byte_pair_ranks = Box::new([[u64::MAX; 256]; 256]);
        let mut id_to_byte = [-1i16; 512];

        let vocab_size = (pair_ranks.len() * 2).max(160000);

        for b in 0..=255 {
            let id = byte_fallback[b] as usize;
            if id < 512 {
                id_to_byte[id] = b as i16;
            }
        }

        for b1 in 0..=255 {
            for b2 in 0..=255 {
                let id1 = byte_fallback[b1];
                let id2 = byte_fallback[b2];
                let pack = ((id1 as u64) << 32) | (id2 as u64);
                if let Some(&val) = pair_ranks.get(&pack) {
                    byte_pair_ranks[b1][b2] = ((val.rank as u64) << 32) | (val.id as u64);
                }
            }
        }

        Self {
            pair_ranks,
            byte_pair_ranks,
            byte_fallback,
            id_to_byte,
            eos_token_id,
            vocab_size,
        }
    }

    #[inline(always)]
    fn get_pair_value(&self, left: u32, right: u32) -> Option<BpeValue> {
        let b1 = if left < 512 {
            unsafe { *self.id_to_byte.get_unchecked(left as usize) }
        } else {
            -1
        };
        let b2 = if right < 512 {
            unsafe { *self.id_to_byte.get_unchecked(right as usize) }
        } else {
            -1
        };

        if b1 >= 0 && b2 >= 0 {
            let packed = unsafe { *self.byte_pair_ranks.get_unchecked(b1 as usize).get_unchecked(b2 as usize) };
            if packed == u64::MAX {
                None
            } else {
                Some(BpeValue {
                    rank: (packed >> 32) as u32,
                    id: packed as u32,
                })
            }
        } else {
            let pack = ((left as u64) << 32) | (right as u64);
            self.pair_ranks.get(&pack).copied()
        }
    }

    pub fn encode_single_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        if len == 1 {
            if *token_count < slice.len() {
                unsafe {
                    *slice.get_unchecked_mut(*token_count) = *self.byte_fallback.get_unchecked(bytes[0] as usize);
                }
                *token_count += 1;
            }
            return;
        }
        if len <= 16 {
            self.encode_short_chunk(bytes, slice, token_count, ctx);
        } else {
            self.encode_long_chunk(bytes, slice, token_count, ctx);
        }
    }

    fn encode_short_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        let prev = unsafe { ctx.short_prev.get_unchecked_mut(..len) };
        let next = unsafe { ctx.short_next.get_unchecked_mut(..len) };
        let ids = unsafe { ctx.short_token_ids.get_unchecked_mut(..len) };

        for i in 0..len {
            unsafe {
                *prev.get_unchecked_mut(i) = i.wrapping_sub(1) as u8;
                *next.get_unchecked_mut(i) = (i + 1) as u8;
                *ids.get_unchecked_mut(i) = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize);
            }
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = usize::MAX;
            let mut best_id = 0u32;

            let mut i = 0;
            while i < len {
                let r = unsafe { *next.get_unchecked(i) as usize };
                if r >= len {
                    break;
                }

                if let Some(val) = unsafe { self.get_pair_value(*ids.get_unchecked(i), *ids.get_unchecked(r)) } {
                    if val.rank < min_rank {
                        min_rank = val.rank;
                        best_id = val.id;
                        best_left = i;
                    }
                }
                i = r;
            }

            if best_left == usize::MAX {
                break;
            }

            let l = best_left;
            let r = unsafe { *next.get_unchecked(l) as usize };
            let after_r = unsafe { *next.get_unchecked(r) };

            unsafe {
                *next.get_unchecked_mut(l) = after_r;
                if (after_r as usize) < len {
                    *prev.get_unchecked_mut(after_r as usize) = l as u8;
                }
                *ids.get_unchecked_mut(l) = best_id;
            }
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            unsafe {
                *slice.get_unchecked_mut(*token_count) = *ids.get_unchecked(i);
                *token_count += 1;
                i = *next.get_unchecked(i) as usize;
            }
        }
    }

    fn encode_long_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        ctx.prev.resize(len * 3, 0);
        ctx.heap.clear();

        for i in 0..len {
            let base = i * 3;
            unsafe {
                *ctx.prev.get_unchecked_mut(base) = *self.byte_fallback.get_unchecked(*bytes.get_unchecked(i) as usize) as usize;
                *ctx.prev.get_unchecked_mut(base + 1) = i.wrapping_sub(1);
                *ctx.prev.get_unchecked_mut(base + 2) = i + 1;
            }
        }

        for i in 0..len - 1 {
            let id1 = unsafe { *ctx.prev.get_unchecked(i * 3) as u32 };
            let id2 = unsafe { *ctx.prev.get_unchecked((i + 1) * 3) as u32 };
            if let Some(val) = self.get_pair_value(id1, id2) {
                // Извлечение и пуш теперь занимают O(1) тактов
                ctx.heap.push(val.rank, i);
            }
        }

        while let Some(pair) = ctx.heap.pop() {
            let left_idx = pair.left_idx;
            let rank = pair.rank;
            let base_l = left_idx * 3;
            let r = unsafe { *ctx.prev.get_unchecked(base_l + 2) };
            if r >= len {
                continue;
            }
            let base_r = r * 3;

            let id_l = unsafe { *ctx.prev.get_unchecked(base_l) as u32 };
            let id_r = unsafe { *ctx.prev.get_unchecked(base_r) as u32 };

            let current_val = self.get_pair_value(id_l, id_r);
            if current_val.map(|v| v.rank) != Some(rank) {
                continue;
            }

            let target_id = current_val.unwrap().id;
            let l_prev = unsafe { *ctx.prev.get_unchecked(base_l + 1) };
            let after_r = unsafe { *ctx.prev.get_unchecked(base_r + 2) };

            unsafe {
                *ctx.prev.get_unchecked_mut(base_l + 2) = after_r;
                if after_r < len {
                    *ctx.prev.get_unchecked_mut(after_r * 3 + 1) = left_idx;
                }
                *ctx.prev.get_unchecked_mut(base_l) = target_id as usize;
            }

            if l_prev != usize::MAX {
                let id_l_prev = unsafe { *ctx.prev.get_unchecked(l_prev * 3) as u32 };
                if let Some(val) = self.get_pair_value(id_l_prev, target_id) {
                    ctx.heap.push(val.rank, l_prev);
                }
            }

            if after_r < len {
                let id_after_r = unsafe { *ctx.prev.get_unchecked(after_r * 3) as u32 };
                if let Some(val) = self.get_pair_value(target_id, id_after_r) {
                    ctx.heap.push(val.rank, left_idx);
                }
            }
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            let base = i * 3;
            unsafe {
                *slice.get_unchecked_mut(*token_count) = *ctx.prev.get_unchecked(base) as u32;
                *token_count += 1;
                i = *ctx.prev.get_unchecked(base + 2);
            }
        }
    }

    pub fn encode_to_pinned(&self, text: &str, pinned_dst: &mut PinnedHostBuffer) -> usize {
        let slice = pinned_dst.as_slice_mut();
        if text.is_empty() {
            if !slice.is_empty() {
                unsafe {
                    *slice.get_unchecked_mut(0) = self.eos_token_id as f32;
                }
                return 1;
            }
            return 0;
        }

        let mut ctx = TokenizationContext::new(self.vocab_size);
        let mut token_count = 0;
        let mut tmp_tokens = [0u32; 1024];

        SimdSplitter::split(text, |chunk| {
            let mut chunk_token_count = 0;
            self.encode_single_chunk(chunk, &mut tmp_tokens, &mut chunk_token_count, &mut ctx);

            let len = slice.len();
            for i in 0..chunk_token_count {
                if token_count < len {
                    unsafe {
                        *slice.get_unchecked_mut(token_count) = *tmp_tokens.get_unchecked(i) as f32;
                    }
                    token_count += 1;
                } else {
                    break;
                }
            }
        });

        token_count
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut tmp_buffer = vec![0u32; text.len() + 1];
        let mut ctx = TokenizationContext::new(self.vocab_size);
        let mut token_count = 0;

        SimdSplitter::split(text, |chunk| {
            let buffer_len = tmp_buffer.len();
            if token_count < buffer_len {
                unsafe {
                    let ptr = tmp_buffer.get_unchecked_mut(token_count..);
                    self.encode_single_chunk(chunk, ptr, &mut token_count, &mut ctx);
                }
            }
        });

        tmp_buffer.truncate(token_count);
        tmp_buffer
    }

    pub fn encode_parallel(&self, texts: &[String]) -> Vec<Vec<u32>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let mut results = vec![Vec::new(); texts.len()];

        let chunk_size = ((texts.len() + num_threads - 1) / num_threads).max(1);
        let mut result_chunks = results.chunks_mut(chunk_size);

        std::thread::scope(|scope| {
            for text_chunk in texts.chunks(chunk_size) {
                if let Some(out_chunk) = result_chunks.next() {
                    scope.spawn(move || {
                        let mut ctx = TokenizationContext::new(self.vocab_size);

                        for (i, text) in text_chunk.iter().enumerate() {
                            if text.is_empty() {
                                continue;
                            }

                            let mut tmp_buffer = vec![0u32; text.len() + 1];
                            let mut token_count = 0;

                            SimdSplitter::split(text, |chunk| {
                                let buffer_len = tmp_buffer.len();
                                if token_count < buffer_len {
                                    unsafe {
                                        let ptr = tmp_buffer.get_unchecked_mut(token_count..);
                                        self.encode_single_chunk(chunk, ptr, &mut token_count, &mut ctx);
                                    }
                                }
                            });

                            tmp_buffer.truncate(token_count);
                            unsafe {
                                *out_chunk.get_unchecked_mut(i) = tmp_buffer;
                            }
                        }
                    });
                }
            }
        });

        results
    }
}

unsafe impl Send for BpeTokenizer {}
unsafe impl Sync for BpeTokenizer {}
