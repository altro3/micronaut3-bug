use super::context::TokenizationContext;
use crate::tokenizer::bpe::dispatcher::BpeEngineDispatcher;
use crate::tokenizer::dfa::runtime::FlatDfaRuntime;
use crate::tokenizer::BpeTokenizer;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TokenizerPipeline {
    pub tokenizer: BpeTokenizer,
    pub dfa_splitter: FlatDfaRuntime,
}

impl TokenizerPipeline {
    pub fn new(merge_table: BpeTokenizer, dfa_splitter: FlatDfaRuntime) -> Self {
        Self {
            tokenizer: merge_table,
            dfa_splitter,
        }
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

    pub fn encode_parallel(&self, texts: &[String], contexts: &mut [TokenizationContext]) -> Vec<Vec<u32>> {
        let total_texts = texts.len();
        let mut results = vec![Vec::new(); total_texts];

        let task_index = AtomicUsize::new(0);
        let results_slice = &mut results[..];
        let results_ref = &*results_slice;

        let mut context_iter = contexts.iter_mut();

        std::thread::scope(|scope| {
            loop {
                let ctx = match context_iter.next() {
                    Some(c) => c,
                    None => break,
                };

                let task_index = &task_index;

                scope.spawn(move || {
                    loop {
                        let idx = task_index.fetch_add(1, Ordering::Relaxed);
                        if idx >= total_texts {
                            break;
                        }

                        let text = unsafe { texts.get_unchecked(idx) };
                        if text.is_empty() {
                            continue;
                        }

                        let tokens = self.encode(text, ctx);

                        unsafe {
                            let out_vec_ptr = (results_ref.as_ptr() as *mut Vec<u32>).add(idx);
                            (*out_vec_ptr).reserve_exact(tokens.len());
                            std::ptr::copy_nonoverlapping(tokens.as_ptr(), (*out_vec_ptr).as_mut_ptr(), tokens.len());
                            (*out_vec_ptr).set_len(tokens.len());
                        }
                    }
                });
            }
        });

        results
    }
}
