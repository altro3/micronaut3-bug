use super::context::TokenizationContext;
use crate::tokenizer::bpe::dispatcher::{
    BpeEngineDispatcher, CACHE_HITS, CACHE_MISSES, CALL_COUNT, FALLBACK_CYCLES, LONG_CYCLES, SHORT_CYCLES, TOTAL_BYTES_PROCESSED,
};
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

    pub fn encode_parallel(&self, texts: &[&str], contexts: &mut [TokenizationContext]) -> Vec<Vec<u32>> {
        let total_texts = texts.len();
        let mut results = vec![Vec::new(); total_texts];

        let task_index = AtomicUsize::new(0);

        let results_ptr = results.as_mut_ptr() as usize;
        let self_ref = self;

        std::thread::scope(|scope| {
            for ctx in contexts.iter_mut() {
                let task_index = &task_index;

                scope.spawn(move || {
                    let local_results_ptr = results_ptr as *mut Vec<u32>;

                    loop {
                        let idx = task_index.fetch_add(1, Ordering::Relaxed);
                        if idx >= total_texts {
                            break;
                        }

                        let text = unsafe { *texts.get_unchecked(idx) };
                        if text.is_empty() {
                            continue;
                        }

                        let tokens = self_ref.encode(text, ctx);

                        unsafe {
                            let out_vec_ptr = local_results_ptr.add(idx);
                            (*out_vec_ptr).reserve_exact(tokens.len());
                            std::ptr::copy_nonoverlapping(tokens.as_ptr(), (*out_vec_ptr).as_mut_ptr(), tokens.len());
                            (*out_vec_ptr).set_len(tokens.len());
                        }
                    }

                    let calls = CALL_COUNT.get();
                    let fb = FALLBACK_CYCLES.get();
                    let short = SHORT_CYCLES.get();
                    let long = LONG_CYCLES.get();
                    let bytes = TOTAL_BYTES_PROCESSED.get();

                    let hits = CACHE_HITS.get();
                    let misses = CACHE_MISSES.get();

                    let total_cycles = fb + short + long;

                    if calls > 0 && total_cycles > 0 {
                        let avg_len = bytes as f64 / calls as f64;
                        let total_cached_lookups = hits + misses;
                        let hit_rate = if total_cached_lookups > 0 {
                            (hits as f64 / total_cached_lookups as f64) * 100.0
                        } else {
                            0.0
                        };

                        println!(
                            "[ПОТОК] Вызовов: {} | СрДлина: {:.1}б | Такты -> FB: {:.1}% | Short: {:.1}% | Long: {:.1}%",
                            calls,
                            avg_len,
                            (fb as f64 / total_cycles as f64) * 100.0,
                            (short as f64 / total_cycles as f64) * 100.0,
                            (long as f64 / total_cycles as f64) * 100.0
                        );
                        println!(
                            "        [КЭШ КОНТЕКСТА] Всего лукапов: {} | Попадания (HitRate): {:.1}% | Промахи (Ушли в словарь): {}",
                            total_cached_lookups, hit_rate, misses
                        );
                    }
                });
            }
        });

        results
    }
}
