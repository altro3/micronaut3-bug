use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;
use crate::cuda::PinnedHostBuffer;
use crate::tokenizer::SimdSplitter;
use std::arch::x86_64::*;

impl BpeTokenizer {
    #[inline(always)]
    pub fn encode<'a>(&self, text: &str, ctx: &'a mut TokenizationContext) -> &'a [u32] {
        if text.is_empty() {
            return &[];
        }

        let text_bytes = text.as_bytes();
        ctx.reset(text_bytes.len());

        let tokens_found = SimdSplitter::split(text, &mut ctx.tokens_buffer, &mut ctx.tokens_lens_buffer);
        if tokens_found == 0 {
            return &[];
        }

        let mut token_count = 0;
        self.encode_long_chunk(text_bytes, &mut token_count, ctx);

        unsafe { ctx.tokens_buffer.get_unchecked(..token_count) }
    }

    pub fn encode_to_pinned(&self, text: &str, pinned_dst: &mut PinnedHostBuffer, ctx: &mut TokenizationContext) -> usize {
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
        ctx.reset(text_bytes.len());

        let tokens_found = SimdSplitter::split(text, &mut ctx.tokens_buffer, &mut ctx.tokens_lens_buffer);
        if tokens_found == 0 {
            return 0;
        }

        let mut token_count = 0;
        self.encode_long_chunk(text_bytes, &mut token_count, ctx);

        let tokens_to_copy = token_count.min(slice_len);

        unsafe {
            let dst_ptr = slice.as_mut_ptr();
            let src_ptr = ctx.tokens_buffer.as_ptr();
            let mut offset = 0;

            while offset + 8 <= tokens_to_copy {
                let chunk = _mm256_loadu_si256(src_ptr.add(offset) as *const __m256i);
                let floats = _mm256_cvtepi32_ps(chunk);

                _mm256_storeu_ps(dst_ptr.add(offset), floats);
                offset += 8;
            }

            while offset < tokens_to_copy {
                let token_id = *src_ptr.add(offset);
                *dst_ptr.add(offset) = token_id as f32;
                offset += 1;
            }
        }

        tokens_to_copy
    }

    pub fn encode_parallel(&self, texts: &[String]) -> Vec<Vec<u32>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let total_texts = texts.len();

        let mut results = Vec::with_capacity(total_texts);
        results.resize_with(total_texts, Vec::new);

        let base_results_ptr = results.as_mut_ptr();

        struct UnsafePtrWrapper(*mut Vec<u32>);
        unsafe impl Send for UnsafePtrWrapper {}
        unsafe impl Sync for UnsafePtrWrapper {}
        let shared_results = UnsafePtrWrapper(base_results_ptr);

        use std::sync::atomic::{AtomicUsize, Ordering};
        let task_index = AtomicUsize::new(0);
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

        std::thread::scope(|scope| {
            for _ in 0..num_threads {
                let shared_results = &shared_results;
                let task_index = &task_index;

                scope.spawn(move || {
                    let mut ctx = TokenizationContext::new(self.vocab_size, 4096);

                    loop {
                        let idx = task_index.fetch_add(1, Ordering::Relaxed);
                        if idx >= total_texts {
                            break;
                        }

                        let text = unsafe { texts.get_unchecked(idx) };
                        if text.is_empty() {
                            continue;
                        }

                        let text_bytes = text.as_bytes();
                        ctx.reset(text_bytes.len());

                        let tokens_found = SimdSplitter::split(text, &mut ctx.tokens_buffer, &mut ctx.tokens_lens_buffer);
                        if tokens_found == 0 {
                            continue;
                        }

                        let mut token_count = 0;
                        self.encode_long_chunk(text_bytes, &mut token_count, &mut ctx);

                        let mut local_res = Vec::with_capacity(token_count);
                        unsafe {
                            std::ptr::copy_nonoverlapping(ctx.tokens_buffer.as_ptr(), local_res.as_mut_ptr(), token_count);
                            local_res.set_len(token_count);
                        }

                        unsafe {
                            let target_res_ptr = shared_results.0.add(idx);
                            std::ptr::write(target_res_ptr, local_res);
                        }
                    }
                });
            }
        });

        results
    }
}
