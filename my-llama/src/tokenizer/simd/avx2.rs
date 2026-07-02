use crate::process_mask_generic;
use crate::tokenizer::simd::utils::consume_tail;
use std::arch::x86_64::*;

#[repr(align(32))]
struct AlignedLookup {
    data: [i8; 32],
}

static LOOKUP_MASK: AlignedLookup = AlignedLookup {
    data: [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    ],
};

#[target_feature(enable = "avx2")]
pub unsafe fn split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut idx = 0;

    let buffer_len = ids_buffer.len();
    let buf_start_ptr = ids_buffer.as_mut_ptr();
    let mut write_ptr = buf_start_ptr;
    let end_write_ptr = unsafe {write_ptr.add(buffer_len) };

    let fallback_ptr = byte_fallback.as_ptr();
    let bytes_ptr = bytes.as_ptr();

    let lookup_mask = unsafe {_mm256_load_si256((&LOOKUP_MASK.data as *const [i8; 32]) as *const __m256i) };
    let low_nibble_mask = _mm256_set1_epi8(0x0F);
    let non_printable_mask = _mm256_set1_epi8(33);

    while idx + 128 <= len {
        let base_ptr = unsafe { bytes_ptr.add(idx) };

        let chunk0 = unsafe { _mm256_loadu_si256(base_ptr as *const __m256i) };
        let chunk1 = unsafe {_mm256_loadu_si256(base_ptr.add(32) as *const __m256i) };
        let chunk2 = unsafe {_mm256_loadu_si256(base_ptr.add(64) as *const __m256i) };
        let chunk3 = unsafe {_mm256_loadu_si256(base_ptr.add(96) as *const __m256i) };

        let delim0 = _mm256_and_si256(
            _mm256_shuffle_epi8(lookup_mask, _mm256_and_si256(chunk0, low_nibble_mask)),
            _mm256_cmpgt_epi8(non_printable_mask, chunk0),
        );
        let delim1 = _mm256_and_si256(
            _mm256_shuffle_epi8(lookup_mask, _mm256_and_si256(chunk1, low_nibble_mask)),
            _mm256_cmpgt_epi8(non_printable_mask, chunk1),
        );
        let delim2 = _mm256_and_si256(
            _mm256_shuffle_epi8(lookup_mask, _mm256_and_si256(chunk2, low_nibble_mask)),
            _mm256_cmpgt_epi8(non_printable_mask, chunk2),
        );
        let delim3 = _mm256_and_si256(
            _mm256_shuffle_epi8(lookup_mask, _mm256_and_si256(chunk3, low_nibble_mask)),
            _mm256_cmpgt_epi8(non_printable_mask, chunk3),
        );

        let m0 = !_mm256_movemask_epi8(delim0) as u32;
        let m1 = !_mm256_movemask_epi8(delim1) as u32;
        let m2 = !_mm256_movemask_epi8(delim2) as u32;
        let m3 = !_mm256_movemask_epi8(delim3) as u32;

        process_mask_generic!(m0, 0, 0, write_ptr, end_write_ptr, base_ptr, fallback_ptr, buffer_len, u32);
        process_mask_generic!(m1, 32, 0, write_ptr, end_write_ptr, base_ptr, fallback_ptr, buffer_len, u32);
        process_mask_generic!(m2, 64, 0, write_ptr, end_write_ptr, base_ptr, fallback_ptr, buffer_len, u32);
        process_mask_generic!(m3, 96, 0, write_ptr, end_write_ptr, base_ptr, fallback_ptr, buffer_len, u32);

        idx += 128;
    }

    consume_tail(
        bytes_ptr,
        len,
        &mut idx,
        &mut write_ptr,
        end_write_ptr,
        fallback_ptr,
        buf_start_ptr,
        buffer_len,
    )
}
