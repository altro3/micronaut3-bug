use super::context::TokenizationContext;
use super::engine_long::LongBpeEngine;
use super::pipeline::TokenizerPipeline;
use crate::cuda::PinnedHostBuffer;
use std::arch::x86_64::*;

impl TokenizerPipeline {
    pub fn encode_to_pinned(&self, text: &str, pinned_dst: &mut PinnedHostBuffer, ctx: &mut TokenizationContext) -> usize {
        let slice = pinned_dst.as_slice_mut();
        let slice_len = slice.len();

        if text.is_empty() {
            if slice_len > 0 {
                unsafe {
                    *slice.get_unchecked_mut(0) = self.tokenizer.eos_token_id as f32;
                }
                return 1;
            }
            return 0;
        }

        let bytes = text.as_bytes();
        ctx.reset_all(bytes.len());

        let tokens_found = self
            .dfa_splitter
            .split_streaming(bytes, &mut ctx.chunk_offsets, &mut ctx.tokens_lens_buffer);
        let mut token_count = 0;

        let offsets_ptr = ctx.chunk_offsets.as_ptr();
        let lengths_ptr = ctx.tokens_lens_buffer.as_ptr();

        for i in 0..tokens_found {
            unsafe {
                let offset = *offsets_ptr.add(i) as usize;
                let length = *lengths_ptr.add(i) as usize;
                LongBpeEngine::merge(&self.tokenizer, &bytes[offset..offset + length], &mut token_count, ctx);
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
}
