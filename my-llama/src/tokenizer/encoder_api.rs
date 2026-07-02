use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;
use super::simd_splitter::{SimdSplitter, TokenSpan};
use crate::cuda::PinnedHostBuffer;

impl BpeTokenizer {
    #[inline(always)]
    pub fn encode<'a>(&self, text: &str, ctx: &'a mut TokenizationContext) -> &'a [u32] {
        if text.is_empty() {
            return &[];
        }

        let text_bytes = text.as_bytes();
        let text_len = text_bytes.len();

        ctx.reset(text_len);

        let mut spans_count = SimdSplitter::split(text, &mut ctx.spans_buffer);

        if spans_count == ctx.spans_buffer.len() {
            ctx.spans_buffer.resize(text_len + 1, TokenSpan { start: 0, end: 0 });
            spans_count = SimdSplitter::split(text, &mut ctx.spans_buffer);
        }

        let mut token_count = 0;

        for i in 0..spans_count {
            let span = unsafe { *ctx.spans_buffer.get_unchecked(i) };
            let chunk = unsafe { text_bytes.get_unchecked(span.start as usize..span.end as usize) };
            self.encode_single_chunk(chunk, &mut ctx.tokens_buffer, &mut token_count, &mut ctx);
        }

        unsafe { ctx.tokens_buffer.get_unchecked(..token_count) }
    }

    pub fn encode_to_pinned(&self, text: &str, pinned_dst: &mut PinnedHostBuffer) -> usize {
        let slice = pinned_dst.as_slice_mut();
        let slice_len = slice.len();

        if text.is_empty() {
            if slice_len > 0 {
                unsafe {
                    *slice.get_unchecked_mut(0) = self.eos_token_id as f32;
                }
                return 1;
            }
            return 0;
        }

        let text_bytes = text.as_bytes();
        let text_len = text_bytes.len();
        let max_possible_tokens = text_len + 1;

        let mut spans_buffer = vec![TokenSpan { start: 0, end: 0 }; (text_len / 2).max(16)];
        let mut spans_count = SimdSplitter::split(text, &mut spans_buffer);

        if spans_count == spans_buffer.len() && spans_buffer.len() < max_possible_tokens {
            spans_buffer.resize(max_possible_tokens, TokenSpan { start: 0, end: 0 });
            spans_count = SimdSplitter::split(text, &mut spans_buffer);
        }

        let mut ctx = TokenizationContext::new(self.vocab_size);
        let mut token_count = 0;
        let mut chunk_tokens = [0u32; 1024];

        for i in 0..spans_count {
            if token_count >= slice_len {
                break;
            }

            let span = unsafe { *spans_buffer.get_unchecked(i) };
            let chunk = unsafe { text_bytes.get_unchecked(span.start as usize..span.end as usize) };
            let mut chunk_token_count = 0;

            self.encode_single_chunk(chunk, &mut chunk_tokens, &mut chunk_token_count, &mut ctx);

            let tokens_to_copy = chunk_token_count.min(slice_len - token_count);
            let src_ptr = chunk_tokens.as_ptr();

            unsafe {
                let dst_ptr = slice.as_mut_ptr().add(token_count);
                for offset in 0..tokens_to_copy {
                    let token_id = *src_ptr.add(offset);
                    *dst_ptr.add(offset) = token_id as f32;
                }
            }

            token_count += tokens_to_copy;
        }

        token_count
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
                        let mut thread_tokens_buffer = Vec::with_capacity(512);
                        let mut thread_spans_buffer = vec![TokenSpan { start: 0, end: 0 }; 256];

                        for (i, text) in text_chunk.iter().enumerate() {
                            if text.is_empty() {
                                continue;
                            }

                            let text_bytes = text.as_bytes();
                            let text_len = text_bytes.len();
                            let max_possible_tokens = text_len + 1;

                            thread_tokens_buffer.resize(max_possible_tokens, 0u32);

                            let required_spans_capacity = (text_len / 2).max(16);
                            if thread_spans_buffer.len() < required_spans_capacity {
                                thread_spans_buffer.resize(required_spans_capacity, TokenSpan { start: 0, end: 0 });
                            }

                            let mut spans_count = SimdSplitter::split(text, &mut thread_spans_buffer);
                            if spans_count == thread_spans_buffer.len() && thread_spans_buffer.len() < max_possible_tokens {
                                thread_spans_buffer.resize(max_possible_tokens, TokenSpan { start: 0, end: 0 });
                                spans_count = SimdSplitter::split(text, &mut thread_spans_buffer);
                            }

                            let mut token_count = 0;

                            for j in 0..spans_count {
                                let span = unsafe { *thread_spans_buffer.get_unchecked(j) };
                                let chunk = unsafe { text_bytes.get_unchecked(span.start as usize..span.end as usize) };
                                let buffer_len = thread_tokens_buffer.len();

                                if token_count < buffer_len {
                                    unsafe {
                                        let ptr = thread_tokens_buffer.get_unchecked_mut(token_count..);
                                        self.encode_single_chunk(chunk, ptr, &mut token_count, &mut ctx);
                                    }
                                } else {
                                    break;
                                }
                            }

                            let mut final_tokens = vec![0u32; token_count];
                            unsafe {
                                std::ptr::copy_nonoverlapping(thread_tokens_buffer.as_ptr(), final_tokens.as_mut_ptr(), token_count);
                                *out_chunk.get_unchecked_mut(i) = final_tokens;
                            }
                        }
                    });
                }
            }
        });

        results
    }
}
