use super::context::TokenizationContext;
use crate::tokenizer::BpeTokenizer;
use std::arch::x86_64::*;

pub struct ShortBpeEngine;

impl ShortBpeEngine {
    #[inline(always)]
    pub fn merge(data: &BpeTokenizer, bytes: &[u8], token_count: &mut usize, ctx: &mut TokenizationContext) {
        let mut len = bytes.len();
        if len == 0 {
            return;
        }

        let mut ids = [0u32; 32];
        let mut next = [0u16; 32];
        let mut prev = [0u16; 32];

        #[repr(align(32))]
        struct AlignedRanks([u32; 32]);
        let mut ranks = AlignedRanks([u32::MAX; 32]);

        unsafe {
            let cache_ptr = ctx.bpe_direct_cache.as_mut_ptr();

            // Оптимальный L2-Friendly кэш под маску 0x7FFF (32768 слотов = 512КБ)
            macro_rules! get_pair_cached {
                ($left:expr, $right:expr) => {{
                    let l = $left as u64;
                    let r = $right as u64;
                    let pack = (l << 32) | r;

                    let hash = pack.wrapping_mul(0x9E3779B97F4A7C15);
                    let cache_idx = ((hash ^ (hash >> 28)) & 0x7FFF) as usize;
                    let slot_ptr = cache_ptr.add(cache_idx);

                    if (*slot_ptr).key == pack {
                        crate::tokenizer::bpe::dispatcher::CACHE_HITS.with(|c| c.set(c.get() + 1));
                        (*slot_ptr).val
                    } else {
                        crate::tokenizer::bpe::dispatcher::CACHE_MISSES.with(|c| c.set(c.get() + 1));
                        let res = data.get_pair_packed($left, $right);
                        (*slot_ptr).key = pack;
                        (*slot_ptr).val = res;
                        res
                    }
                }};
            }

            let fallback_ptr = data.byte_fallback.as_ptr();
            for i in 0..len {
                let b = *bytes.get_unchecked(i) as usize;
                *ids.get_unchecked_mut(i) = *fallback_ptr.add(b);
                *next.get_unchecked_mut(i) = if i == len - 1 { 0xFFFF } else { (i + 1) as u16 };
                *prev.get_unchecked_mut(i) = if i == 0 { 0xFFFF } else { (i - 1) as u16 };
            }

            for i in 0..(len - 1) {
                let packed = get_pair_cached!(*ids.get_unchecked(i), *ids.get_unchecked(i + 1));
                *ranks.0.get_unchecked_mut(i) = if packed != u64::MAX { (packed >> 32) as u32 } else { u32::MAX };
            }

            loop {
                let (min_rank, best_left) = if len <= 9 {
                    // --- ДИНАМИЧЕСКИЙ ПУТЬ 1: Ультра-короткие чанки (основной поток кириллицы Qwen) ---
                    // Скалярный развернутый поиск минимума. Компилятор положит это в регистры общего назначения.
                    let mut m_rank = u32::MAX;
                    let mut b_left = 0xFFFFusize;
                    let mut i = 0usize;

                    loop {
                        let r = *ranks.0.get_unchecked(i);
                        if r < m_rank {
                            m_rank = r;
                            b_left = i;
                        }
                        let n = *next.get_unchecked(i);
                        if n == 0xFFFF {
                            break;
                        }
                        i = n as usize;
                    }
                    (m_rank, b_left)
                } else if len <= 17 {
                    // --- ДИНАМИЧЕСКИЙ ПУТЬ 2: Средние чанки (до 16 рангов) ---
                    // Используем ровно два AVX2 регистра, никаких тяжелых 4-регистровых пермутаций
                    let r0 = _mm256_load_si256(ranks.0.as_ptr() as *const __m256i);
                    let r1 = _mm256_load_si256(ranks.0.as_ptr().add(8) as *const __m256i);

                    let min_all = _mm256_min_epu32(r0, r1);
                    let mut tmp = _mm_min_epu32(_mm256_castsi256_si128(min_all), _mm256_extractf128_si256(min_all, 1));
                    tmp = _mm_min_epu32(tmp, _mm_shuffle_epi32(tmp, 0x4E));
                    tmp = _mm_min_epu32(tmp, _mm_shuffle_epi32(tmp, 0xB1));
                    let m_rank = _mm_cvtsi128_si32(tmp) as u32;

                    if m_rank == u32::MAX {
                        (u32::MAX, 0xFFFFusize)
                    } else {
                        let v_min = _mm256_set1_epi32(m_rank as i32);
                        let m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(r0, v_min)) as u32;
                        let m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(r1, v_min)) as u32;

                        let b_left = if m0 != 0 {
                            (m0.trailing_zeros() >> 2) as usize
                        } else {
                            8 + (m1.trailing_zeros() >> 2) as usize
                        };
                        (m_rank, b_left)
                    }
                } else {
                    // --- ДИНАМИЧЕСКИЙ ПУТЬ 3: Полный AVX2 (только для длинных кусков) ---
                    let r0 = _mm256_load_si256(ranks.0.as_ptr() as *const __m256i);
                    let r1 = _mm256_load_si256(ranks.0.as_ptr().add(8) as *const __m256i);
                    let r2 = _mm256_load_si256(ranks.0.as_ptr().add(16) as *const __m256i);
                    let r3 = _mm256_load_si256(ranks.0.as_ptr().add(24) as *const __m256i);

                    let min_01 = _mm256_min_epu32(r0, r1);
                    let min_23 = _mm256_min_epu32(r2, r3);
                    let min_all = _mm256_min_epu32(min_01, min_23);

                    let mut tmp = _mm256_min_epu32(min_all, _mm256_permute2f128_si256(min_all, min_all, 1));
                    tmp = _mm256_min_epu32(tmp, _mm256_shuffle_epi32(tmp, 0x4E));
                    tmp = _mm256_min_epu32(tmp, _mm256_shuffle_epi32(tmp, 0xB1));
                    let m_rank = _mm_cvtsi128_si32(_mm256_castsi256_si128(tmp)) as u32;

                    if m_rank == u32::MAX {
                        (u32::MAX, 0xFFFFusize)
                    } else {
                        let v_min = _mm256_set1_epi32(m_rank as i32);
                        let m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(r0, v_min)) as u32;
                        let m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(r1, v_min)) as u32;
                        let m2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(r2, v_min)) as u32;
                        let m3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(r3, v_min)) as u32;

                        let b_left = if m0 != 0 {
                            (m0.trailing_zeros() >> 2) as usize
                        } else if m1 != 0 {
                            8 + (m1.trailing_zeros() >> 2) as usize
                        } else if m2 != 0 {
                            16 + (m2.trailing_zeros() >> 2) as usize
                        } else {
                            24 + (m3.trailing_zeros() >> 2) as usize
                        };
                        (m_rank, b_left)
                    }
                };

                if min_rank == u32::MAX {
                    break;
                }

                let right_idx = *next.get_unchecked(best_left) as usize;
                let packed = get_pair_cached!(*ids.get_unchecked(best_left), *ids.get_unchecked(right_idx));

                *ids.get_unchecked_mut(best_left) = packed as u32;
                let after_r = *next.get_unchecked(right_idx);
                *next.get_unchecked_mut(best_left) = after_r;

                if after_r != 0xFFFF {
                    *prev.get_unchecked_mut(after_r as usize) = best_left as u16;
                }
                *ranks.0.get_unchecked_mut(right_idx) = u32::MAX;

                if after_r != 0xFFFF {
                    let packed_r = get_pair_cached!(*ids.get_unchecked(best_left), *ids.get_unchecked(after_r as usize));
                    *ranks.0.get_unchecked_mut(best_left) = if packed_r != u64::MAX { (packed_r >> 32) as u32 } else { u32::MAX };
                } else {
                    *ranks.0.get_unchecked_mut(best_left) = u32::MAX;
                }

                let p_idx = *prev.get_unchecked(best_left);
                if p_idx != 0xFFFF {
                    let p = p_idx as usize;
                    let packed_l = get_pair_cached!(*ids.get_unchecked(p), *ids.get_unchecked(best_left));
                    *ranks.0.get_unchecked_mut(p) = if packed_l != u64::MAX { (packed_l >> 32) as u32 } else { u32::MAX };
                }
            }

            let mut curr = 0usize;
            let mut count = *token_count;
            let out_ptr = ctx.tokens_buffer.as_mut_ptr();
            loop {
                *out_ptr.add(count) = *ids.get_unchecked(curr);
                count += 1;
                let n = *next.get_unchecked(curr);
                if n == 0xFFFF {
                    break;
                }
                curr = n as usize;
            }
            *token_count = count;
            ctx.tokens_buffer.set_len(count);
        }
    }
}
