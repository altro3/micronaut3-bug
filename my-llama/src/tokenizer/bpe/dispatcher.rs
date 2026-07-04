use super::context::TokenizationContext;
use super::engine_long::LongBpeEngine;
use super::engine_short::ShortBpeEngine;
use crate::tokenizer::BpeTokenizer;

pub struct BpeEngineDispatcher;

impl BpeEngineDispatcher {
    #[inline(always)]
    pub fn merge(data: &BpeTokenizer, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let len = bytes.len();
        if len == 0 {
            return;
        }

        if len == 1 {
            unsafe {
                let b = *bytes.get_unchecked(0) as usize;
                let token_id = *data.byte_fallback.get_unchecked(b);
                *ctx.tokens_buffer.as_mut_ptr().add(*token_count) = token_id;
                *token_count += 1;
                ctx.tokens_buffer.set_len(*token_count);
            }
            return;
        }

        if len <= 32 {
            ShortBpeEngine::merge(data, bytes, token_count, ctx);
        } else {
            LongBpeEngine::merge(data, bytes, token_count, ctx);
        }
    }
}
