use super::context::TokenizationContext;
use crate::cuda::PinnedHostBuffer;
use crate::token::bpe_types::BpePair;
use rustc_hash::FxHashMap;

pub struct BpeTokenizer {
    pair_ranks: FxHashMap<u64, u32>,
    byte_pair_ranks: Box<[[u32; 256]; 256]>,
    byte_fallback: [u32; 256],
    id_to_byte: [i16; 512],
    pub eos_token_id: u32,
}

impl BpeTokenizer {
    pub fn new(pair_ranks: FxHashMap<u64, u32>, byte_fallback: [u32; 256], eos_token_id: u32) -> Self {
        let mut byte_pair_ranks = Box::new([[u32::MAX; 256]; 256]);
        let mut id_to_byte = [-1i16; 512];

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
                if let Some(&rank) = pair_ranks.get(&pack) {
                    byte_pair_ranks[b1][b2] = rank;
                }
            }
        }

        Self {
            pair_ranks,
            byte_pair_ranks,
            byte_fallback,
            id_to_byte,
            eos_token_id,
        }
    }

    #[inline(always)]
    fn get_pair_rank(&self, left: u32, right: u32) -> Option<u32> {
        let b1 = if (left as usize) < 512 { self.id_to_byte[left as usize] } else { -1 };
        let b2 = if (right as usize) < 512 { self.id_to_byte[right as usize] } else { -1 };

        if b1 >= 0 && b2 >= 0 {
            let rank = self.byte_pair_ranks[b1 as usize][b2 as usize];
            if rank == u32::MAX { None } else { Some(rank) }
        } else {
            let pack = ((left as u64) << 32) | (right as u64);
            self.pair_ranks.get(&pack).copied()
        }
    }

    #[inline(always)]
    fn tokenize_chunks(&self, text: &str, mut f: impl for<'b> FnMut(&'b [u8])) {
        static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re = RE.get_or_init(|| regex::Regex::new(r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]+|\p{L}+|\p{N}+|\s+(?!\S)|\s+"#).unwrap());
        for mat in re.find_iter(text) {
            f(mat.as_str().as_bytes());
        }
    }

    pub fn encode_single_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        if len == 1 {
            if *token_count < slice.len() {
                slice[*token_count] = self.byte_fallback[bytes[0] as usize];
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
        let prev = &mut ctx.short_prev[..len];
        let next = &mut ctx.short_next[..len];
        let ids = &mut ctx.short_token_ids[..len];

        for i in 0..len {
            prev[i] = i.wrapping_sub(1) as u8;
            next[i] = (i + 1) as u8;
            ids[i] = self.byte_fallback[bytes[i] as usize];
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut best_left = usize::MAX;
            let mut i = 0;
            while i < len {
                let r = next[i] as usize;
                if r >= len {
                    break;
                }
                if let Some(rank) = self.get_pair_rank(ids[i], ids[r]) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_left = i;
                    }
                }
                i = r;
            }
            if best_left == usize::MAX {
                break;
            }
            let l = best_left;
            let r = next[l] as usize;
            let after_r = next[r];
            next[l] = after_r;
            if (after_r as usize) < len {
                prev[after_r as usize] = l as u8;
            }
            ids[l] = min_rank;
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            slice[*token_count] = ids[i];
            *token_count += 1;
            i = next[i] as usize;
        }
    }

    fn encode_long_chunk(&self, bytes: &[u8], slice: &mut [u32], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        ctx.prev.resize(len * 3, 0);
        ctx.heap.clear();

        for i in 0..len {
            let base = i * 3;
            ctx.prev[base] = self.byte_fallback[bytes[i] as usize] as usize;
            ctx.prev[base + 1] = i.wrapping_sub(1);
            ctx.prev[base + 2] = i + 1;
        }

        for i in 0..len - 1 {
            let id1 = ctx.prev[i * 3] as u32;
            let id2 = ctx.prev[(i + 1) * 3] as u32;
            if let Some(rank) = self.get_pair_rank(id1, id2) {
                ctx.heap.push(BpePair { rank, left_idx: i });
            }
        }

        while let Some(BpePair { rank, left_idx }) = ctx.heap.pop() {
            let base_l = left_idx * 3;
            let r = ctx.prev[base_l + 2];
            if r >= len {
                continue;
            }
            let base_r = r * 3;

            let id_l = ctx.prev[base_l] as u32;
            let id_r = ctx.prev[base_r] as u32;

            if self.get_pair_rank(id_l, id_r) != Some(rank) {
                continue;
            }

            let l_prev = ctx.prev[base_l + 1];
            let after_r = ctx.prev[base_r + 2];

            ctx.prev[base_l + 2] = after_r;
            if after_r < len {
                ctx.prev[after_r * 3 + 1] = left_idx;
            }

            ctx.prev[base_l] = rank as usize;

            if l_prev != usize::MAX {
                let id_l_prev = ctx.prev[l_prev * 3] as u32;
                if let Some(r_new) = self.get_pair_rank(id_l_prev, rank) {
                    ctx.heap.push(BpePair {
                        rank: r_new,
                        left_idx: l_prev,
                    });
                }
            }

            if after_r < len {
                let id_after_r = ctx.prev[after_r * 3] as u32;
                if let Some(r_new) = self.get_pair_rank(rank, id_after_r) {
                    ctx.heap.push(BpePair { rank: r_new, left_idx });
                }
            }
        }

        let mut i = 0;
        while i < len {
            if *token_count >= slice.len() {
                break;
            }
            let base = i * 3;
            slice[*token_count] = ctx.prev[base] as u32;
            *token_count += 1;
            i = ctx.prev[base + 2];
        }
    }

    pub fn encode_to_pinned(&self, text: &str, pinned_dst: &mut PinnedHostBuffer) -> usize {
        let slice = pinned_dst.as_slice_mut();
        if text.is_empty() {
            if !slice.is_empty() {
                slice[0] = self.eos_token_id as f32;
                return 1;
            }
            return 0;
        }

        let mut ctx = TokenizationContext::new();
        let mut token_count = 0;

        let mut tmp_tokens = [0u32; 1024];

        self.tokenize_chunks(text, |chunk| {
            let mut chunk_token_count = 0;
            self.encode_single_chunk(chunk, &mut tmp_tokens, &mut chunk_token_count, &mut ctx);
            for i in 0..chunk_token_count {
                if token_count < slice.len() {
                    slice[token_count] = tmp_tokens[i] as f32;
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
        let mut ctx = TokenizationContext::new();
        let mut token_count = 0;

        self.tokenize_chunks(text, |chunk| {
            if token_count < tmp_buffer.len() {
                self.encode_single_chunk(chunk, &mut tmp_buffer[token_count..], &mut token_count, &mut ctx);
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
        let chunk_size = (texts.len() + num_threads - 1) / num_threads;
        let mut result_chunks = results.chunks_mut(chunk_size);

        std::thread::scope(|scope| {
            for text_chunk in texts.chunks(chunk_size) {
                let out_chunk = result_chunks.next().unwrap();

                scope.spawn(move || {
                    let mut ctx = TokenizationContext::new();

                    for (i, text) in text_chunk.iter().enumerate() {
                        if text.is_empty() {
                            continue;
                        }

                        let mut tmp_buffer = vec![0u32; text.len() + 1];
                        let mut token_count = 0;

                        self.tokenize_chunks(text, |chunk| {
                            if token_count < tmp_buffer.len() {
                                self.encode_single_chunk(chunk, &mut tmp_buffer[token_count..], &mut token_count, &mut ctx);
                            }
                        });

                        tmp_buffer.truncate(token_count);
                        out_chunk[i] = tmp_buffer;
                    }
                });
            }
        });

        results
    }
}

unsafe impl Send for BpeTokenizer {}
unsafe impl Sync for BpeTokenizer {}
