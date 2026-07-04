use crate::tokenizer::bpe::context::TokenizationContext;
use crate::tokenizer::dfa::runtime::FlatDfaRuntime;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

pub struct CorpusAggregator;

impl CorpusAggregator {
    pub fn collect_unique_words(text: &str, dfa_splitter: &FlatDfaRuntime, num_threads: usize) -> (Vec<Vec<u32>>, Vec<u32>) {
        let text_bytes = text.as_bytes();
        let chunk_size = (text_bytes.len() + num_threads - 1) / num_threads;

        let mut counts_map: HashMap<Vec<u32>, u32> = HashMap::with_capacity(65536);
        let counts_map_ptr = &mut counts_map as *mut HashMap<Vec<u32>, u32> as usize;

        let spin_lock = std::sync::atomic::AtomicBool::new(false);
        let spin_lock_ptr = &spin_lock as *const std::sync::atomic::AtomicBool as usize;

        let mut contexts: Vec<TokenizationContext> = std::iter::repeat_with(|| TokenizationContext::new(260000, 4096))
            .take(num_threads)
            .collect();

        std::thread::scope(|scope| {
            let counts_map_ref = counts_map_ptr;
            let lock_ref = spin_lock_ptr;

            for (t_idx, ctx) in contexts.iter_mut().enumerate() {
                scope.spawn(move || {
                    let start_pos = (t_idx * chunk_size).min(text_bytes.len());
                    let mut end_pos = ((t_idx + 1) * chunk_size).min(text_bytes.len());
                    while end_pos < text_bytes.len() && !text.is_char_boundary(end_pos) {
                        end_pos += 1;
                    }
                    let local_chunk = &text_bytes[start_pos..end_pos];
                    if local_chunk.is_empty() {
                        return;
                    }

                    let tokens_found = dfa_splitter.split_streaming(local_chunk, &mut ctx.chunk_offsets, &mut ctx.tokens_lens_buffer);

                    let offsets_ptr = ctx.chunk_offsets.as_ptr();
                    let lengths_ptr = ctx.tokens_lens_buffer.as_ptr();
                    let mut local_counts: HashMap<Vec<u32>, u32> = HashMap::with_capacity(4096);

                    for i in 0..tokens_found {
                        unsafe {
                            let offset = *offsets_ptr.add(i) as usize;
                            let length = *lengths_ptr.add(i) as usize;
                            let chunk_bytes = local_chunk.get_unchecked(offset..offset + length);
                            let word_u32: Vec<u32> = chunk_bytes.iter().map(|&b| b as u32).collect();
                            *local_counts.entry(word_u32).or_insert(0) += 1;
                        }
                    }

                    unsafe {
                        let global_map = &mut *(counts_map_ref as *mut HashMap<Vec<u32>, u32>);
                        let atomic_lock = &*(lock_ref as *const std::sync::atomic::AtomicBool);

                        while atomic_lock
                            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                            .is_err()
                        {
                            std::hint::spin_loop();
                        }
                        for (k, v) in local_counts {
                            *global_map.entry(k).or_insert(0) += v;
                        }
                        atomic_lock.store(false, Ordering::Release);
                    }
                });
            }
        });

        println!("[ТРЕНЕР] Фаза агрегации завершена. Уникальных Qwen-цепочек: {}", counts_map.len());

        let (mut global_words, mut global_counts) = (Vec::with_capacity(counts_map.len()), Vec::with_capacity(counts_map.len()));
        for (w_u32, count) in counts_map {
            global_words.push(w_u32);
            global_counts.push(count);
        }

        (global_words, global_counts)
    }
}
