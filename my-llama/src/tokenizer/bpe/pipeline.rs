use super::context::TokenizationContext;
use crate::tokenizer::BpeTokenizer;
use crate::tokenizer::bpe::dispatcher::BpeEngineDispatcher;
use crate::tokenizer::dfa::runtime::FlatDfaRuntime;
use rayon::prelude::*;

pub struct TokenizerPipeline {
    pub tokenizer: BpeTokenizer,
    pub dfa_splitter: FlatDfaRuntime,
}

impl TokenizerPipeline {
    pub fn new(merge_table: BpeTokenizer, dfa_splitter: FlatDfaRuntime) -> Self {
        Self { tokenizer: merge_table, dfa_splitter }
    }

    #[inline(always)]
    pub fn encode<'a>(&self, text: &str, ctx: &'a mut TokenizationContext) -> &'a [u32] {
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
                let chunk_bytes = bytes.get_unchecked(offset..offset + length);

                BpeEngineDispatcher::merge(&self.tokenizer, chunk_bytes, &mut token_count, ctx);
            }
        }

        unsafe { ctx.tokens_buffer.get_unchecked(..token_count) }
    }

    pub fn encode_parallel(&self, texts: &[&str], contexts: &mut [TokenizationContext]) -> Vec<Vec<u32>> {
        let num_contexts = contexts.len();
        let self_ref = self;
        let contexts_ptr = contexts.as_mut_ptr() as usize;

        texts
            .par_iter()
            .enumerate()
            .map_init(
                || rayon::current_thread_index().unwrap_or(0) % num_contexts,
                move |thread_ctx_idx, (_global_idx, text)| {
                    if text.is_empty() {
                        Vec::new()
                    } else {
                        unsafe {
                            let local_contexts_ptr = contexts_ptr as *mut TokenizationContext;
                            let ctx = &mut *local_contexts_ptr.add(*thread_ctx_idx);
                            self_ref.encode(text, ctx).to_vec()
                        }
                    }
                },
            )
            .collect()
    }
}
