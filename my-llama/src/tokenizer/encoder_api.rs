use super::bpe_tokenizer::BpeTokenizer;
use super::context::TokenizationContext;
use crate::cuda::PinnedHostBuffer;
use crate::tokenizer::SimdSplitter;
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicUsize, Ordering};

impl BpeTokenizer {
    #[inline(always)]
    pub fn encode<'a>(&self, text: &str, ctx: &'a mut TokenizationContext) -> &'a [u32] {
        if text.is_empty() {
            return &[];
        }

        let text_bytes = text.as_bytes();
        let text_len = text_bytes.len();

        if text_len + 1 > ctx.tokens_lens_buffer.capacity() {
            let new_cap = (text_len + 1).next_power_of_two();
            ctx.tokens_lens_buffer.reserve_exact(new_cap - ctx.tokens_lens_buffer.len());
            ctx.chunk_offsets.reserve_exact(new_cap - ctx.chunk_offsets.len());
        }

        unsafe {
            ctx.tokens_lens_buffer.set_len(text_len + 1);
            ctx.chunk_offsets.set_len(text_len + 1);
        }

        let tokens_found = SimdSplitter::split(text, &mut ctx.chunk_offsets, &mut ctx.tokens_lens_buffer);
        if tokens_found == 0 {
            return &[];
        }

        let offsets_ptr = ctx.chunk_offsets.as_ptr();
        let lengths_ptr = ctx.tokens_lens_buffer.as_ptr();

        let mut token_count = 0;
        ctx.tokens_buffer.clear();

        if text_len > ctx.tokens_buffer.capacity() {
            ctx.tokens_buffer.reserve(text_len - ctx.tokens_buffer.len());
        }

        for i in 0..tokens_found {
            unsafe {
                let offset = *offsets_ptr.add(i) as usize;
                let length = *lengths_ptr.add(i) as usize;
                let chunk_bytes = &text_bytes[offset..offset + length];

                self.encode_single_chunk(chunk_bytes, &mut token_count, ctx);
            }
        }

        unsafe {
            ctx.tokens_buffer.set_len(token_count);
            ctx.tokens_buffer.get_unchecked(..token_count)
        }
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
        let text_len = text_bytes.len();

        if text_len + 1 > ctx.long_ids.capacity() {
            let new_cap = (text_len + 1).next_power_of_two();
            ctx.tokens_lens_buffer.reserve_exact(new_cap - ctx.tokens_lens_buffer.len());
            ctx.chunk_offsets.reserve_exact(new_cap - ctx.chunk_offsets.len());
        }

        unsafe {
            ctx.tokens_lens_buffer.set_len(text_len + 1);
            ctx.chunk_offsets.set_len(text_len + 1);
        }

        let tokens_found = SimdSplitter::split(text, &mut ctx.chunk_offsets, &mut ctx.tokens_lens_buffer);
        if tokens_found == 0 {
            return 0;
        }

        let offsets_ptr = ctx.chunk_offsets.as_ptr();
        let lengths_ptr = ctx.tokens_lens_buffer.as_ptr();

        let mut token_count = 0;
        ctx.tokens_buffer.clear();

        if text_len > ctx.tokens_buffer.capacity() {
            ctx.tokens_buffer.reserve(text_len - ctx.tokens_buffer.len());
        }

        for i in 0..tokens_found {
            unsafe {
                let offset = *offsets_ptr.add(i) as usize;
                let length = *lengths_ptr.add(i) as usize;
                let chunk_bytes = &text_bytes[offset..offset + length];

                self.encode_single_chunk(chunk_bytes, &mut token_count, ctx);
            }
        }

        let tokens_to_copy = token_count.min(slice_len);

        unsafe {
            let dst_ptr = slice.as_mut_ptr();
            let src_ptr = ctx.tokens_buffer.as_ptr();
            let mut offset = 0;

            while offset + 8 <= tokens_to_copy {
                let chunk = _mm256_loadu_si256(src_ptr.add(offset) as *const __m256i);
                let floats = _mm256_cvtepi32_ps(chunk);

                _mm256_stream_ps(dst_ptr.add(offset), floats);
                offset += 8;
            }

            while offset < tokens_to_copy {
                let token_id = *src_ptr.add(offset);
                *dst_ptr.add(offset) = token_id as f32;
                offset += 1;
            }

            _mm_sfence();
        }

        tokens_to_copy
    }

    pub fn encode_parallel(&self, texts: &[String]) -> Vec<Vec<u32>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let total_texts = texts.len();
        let mut results = vec![Vec::new(); total_texts];
        let mut max_text_len = 0;

        for i in 0..total_texts {
            unsafe {
                let text_len = texts.get_unchecked(i).len();
                if text_len > max_text_len {
                    max_text_len = text_len;
                }
                if text_len > 0 {
                    results.get_unchecked_mut(i).reserve_exact(text_len);
                }
            }
        }

        let optimal_chunk_capacity = (max_text_len + 64).max(512).next_power_of_two();

        let task_index = AtomicUsize::new(0);
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

        let mut contexts: Vec<Box<TokenizationContext>> = (0..num_threads)
            .map(|_| Box::new(TokenizationContext::new(self.vocab_size, optimal_chunk_capacity)))
            .collect();

        let base_address = results.as_mut_ptr() as usize;

        std::thread::scope(|scope| {
            for thread_id in 0..num_threads {
                let task_index = &task_index;
                let base_address = base_address;

                let ctx_ptr = &mut *contexts[thread_id] as *mut TokenizationContext as usize;

                scope.spawn(move || {
                    let ctx = unsafe { &mut *(ctx_ptr as *mut TokenizationContext) };
                    let target_ptr = base_address as *mut Vec<u32>;

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
                        let text_len = text_bytes.len();

                        if text_len + 1 > ctx.tokens_lens_buffer.capacity() {
                            let new_cap = (text_len + 1).next_power_of_two();
                            ctx.tokens_lens_buffer.reserve_exact(new_cap - ctx.tokens_lens_buffer.len());
                            ctx.chunk_offsets.reserve_exact(new_cap - ctx.chunk_offsets.len());
                        }

                        unsafe {
                            ctx.tokens_lens_buffer.set_len(text_len + 1);
                            ctx.chunk_offsets.set_len(text_len + 1);
                        }

                        let tokens_found = SimdSplitter::split(text, &mut ctx.chunk_offsets, &mut ctx.tokens_lens_buffer);
                        if tokens_found == 0 {
                            continue;
                        }

                        let offsets_ptr = ctx.chunk_offsets.as_ptr();
                        let lengths_ptr = ctx.tokens_lens_buffer.as_ptr();

                        let mut token_count = 0;
                        ctx.tokens_buffer.clear();

                        for i in 0..tokens_found {
                            unsafe {
                                let offset = *offsets_ptr.add(i) as usize;
                                let length = *lengths_ptr.add(i) as usize;
                                let chunk_bytes = &text_bytes[offset..offset + length];

                                self.encode_single_chunk(chunk_bytes, &mut token_count, ctx);
                            }
                        }

                        unsafe {
                            let out_vec_ptr = target_ptr.add(idx);
                            (*out_vec_ptr).reserve_exact(token_count);
                            std::ptr::copy_nonoverlapping(ctx.tokens_buffer.as_ptr(), (*out_vec_ptr).as_mut_ptr(), token_count);
                            (*out_vec_ptr).set_len(token_count);
                        }
                    }
                });
            }
        });

        results
    }
}
