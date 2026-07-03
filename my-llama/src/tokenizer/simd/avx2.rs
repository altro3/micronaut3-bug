use crate::tokenizer::simd::utils::consume_tail;
use std::arch::x86_64::*;

#[repr(align(32))]
struct AlignedLookup {
    data: [i8; 32],
}

static LOOKUP_MASK: AlignedLookup = AlignedLookup {
    data: [
        -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0,
    ],
};

#[target_feature(enable = "avx2,bmi2")]
pub fn split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut idx = 0;

    let buffer_len = ids_buffer.len();
    let buf_start_ptr = ids_buffer.as_mut_ptr();
    let mut write_ptr = buf_start_ptr;
    let end_write_ptr = unsafe { write_ptr.add(buffer_len) };

    let fallback_ptr = byte_fallback.as_ptr();

    let lookup_mask = unsafe { _mm256_load_si256((&LOOKUP_MASK.data as *const [i8; 32]) as *const __m256i) };
    let low_nibble_mask = _mm256_set1_epi8(0x0F);

    let non_printable_mask_unsigned = _mm256_set1_epi8((33u8 ^ 0x80u8) as i8);
    let sign_bit = _mm256_set1_epi8(i8::MIN);

    while idx + 32 <= len {
        let base_ptr = unsafe { bytes.as_ptr().add(idx) };
        let chunk = unsafe { _mm256_loadu_si256(base_ptr as *const __m256i) };

        let chunk_unsigned = _mm256_xor_si256(chunk, sign_bit);
        let is_less_than_33 = _mm256_cmpgt_epi8(non_printable_mask_unsigned, chunk_unsigned);

        let delim = _mm256_and_si256(
            _mm256_shuffle_epi8(lookup_mask, _mm256_and_si256(chunk, low_nibble_mask)),
            is_less_than_33,
        );

        let delim_mask = _mm256_movemask_epi8(delim) as u32;
        let mut valid_mask = delim_mask ^ 0xFFFFFFFF;

        while valid_mask != 0 {
            if write_ptr >= end_write_ptr {
                return buffer_len;
            }

            let tz = valid_mask.trailing_zeros() as usize;

            unsafe {
                let byte_val = *base_ptr.add(tz);
                *write_ptr = *fallback_ptr.add(byte_val as usize);
                write_ptr = write_ptr.add(1);
            }

            valid_mask &= valid_mask - 1;
        }

        idx += 32;
    }

    consume_tail(
        bytes.as_ptr(),
        len,
        &mut idx,
        &mut write_ptr,
        end_write_ptr,
        byte_fallback.as_ptr(),
        buf_start_ptr,
        buffer_len,
    )
}
