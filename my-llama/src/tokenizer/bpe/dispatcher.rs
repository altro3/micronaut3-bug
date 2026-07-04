use super::context::TokenizationContext;
use super::engine_long::LongBpeEngine;
use super::engine_short::ShortBpeEngine;
use crate::tokenizer::BpeTokenizer;
use std::cell::Cell;

thread_local! {
    pub static CALL_COUNT: Cell<u64> = const { Cell::new(0) };
    pub static FALLBACK_CYCLES: Cell<u64> = const { Cell::new(0) };
    pub static SHORT_CYCLES: Cell<u64> = const { Cell::new(0) };
    pub static LONG_CYCLES: Cell<u64> = const { Cell::new(0) };
    pub static TOTAL_BYTES_PROCESSED: Cell<u64> = const { Cell::new(0) };
    pub static DFA_CYCLES: Cell<u64> = const { Cell::new(0) };
    pub static DISPATCH_LOOP_CYCLES: Cell<u64> = const { Cell::new(0) };

    pub static CACHE_HITS: Cell<u64> = const { Cell::new(0) };
    pub static CACHE_MISSES: Cell<u64> = const { Cell::new(0) };
}

pub struct BpeEngineDispatcher;

impl BpeEngineDispatcher {
    #[inline(always)]
    pub fn merge(data: &BpeTokenizer, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        CALL_COUNT.with(|c| c.set(c.get() + 1));
        TOTAL_BYTES_PROCESSED.with(|c| c.set(c.get() + len as u64));

        if len == 1 {
            let start = unsafe { std::arch::x86_64::_rdtsc() };
            unsafe {
                let b = *bytes.get_unchecked(0) as usize;
                let token_id = *data.byte_fallback.get_unchecked(b);
                *ctx.tokens_buffer.as_mut_ptr().add(*token_count) = token_id;
                *token_count += 1;
                ctx.tokens_buffer.set_len(*token_count);
            }
            let end = unsafe { std::arch::x86_64::_rdtsc() };
            FALLBACK_CYCLES.with(|c| c.set(c.get() + (end - start)));
            return;
        }

        if len <= 32 {
            let start = unsafe { std::arch::x86_64::_rdtsc() };
            ShortBpeEngine::merge(data, bytes, token_count, ctx);
            let end = unsafe { std::arch::x86_64::_rdtsc() };
            SHORT_CYCLES.with(|c| c.set(c.get() + (end - start)));
        } else {
            let start = unsafe { std::arch::x86_64::_rdtsc() };
            LongBpeEngine::merge(data, bytes, token_count, ctx);
            let end = unsafe { std::arch::x86_64::_rdtsc() };
            LONG_CYCLES.with(|c| c.set(c.get() + (end - start)));
        }
    }
}