use crate::process_mask_generic;
use crate::tokenizer::arch::utils::consume_tail;
use std::arch::x86_64::*;

#[repr(align(64))]
struct AlignedLookup512 {
    data: [i8; 64],
}

static LOOKUP_MASK_512: AlignedLookup512 = AlignedLookup512 {
    data: [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    ],
};

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut idx = 0;

    let buffer_len = ids_buffer.len();
    let mut write_ptr = ids_buffer.as_mut_ptr();
    let end_write_ptr = unsafe { write_ptr.add(buffer_len) };

    let fallback_ptr = byte_fallback.as_ptr();
    let bytes_ptr = bytes.as_ptr();

    let lookup_mask = unsafe { _mm512_load_si512((&LOOKUP_MASK_512.data as *const [i8; 64]) as *const __m512i) };
    let low_nibble_mask = _mm512_set1_epi8(0x0F);
    let non_printable_mask = _mm512_set1_epi8(33);

    while idx + 256 <= len {
        let base_ptr = unsafe { bytes_ptr.add(idx) };

        let chunk0 = unsafe { _mm512_loadu_si512(base_ptr as *const __m512i) };
        let chunk1 = unsafe { _mm512_loadu_si512(base_ptr.add(64) as *const __m512i) };
        let chunk2 = unsafe { _mm512_loadu_si512(base_ptr.add(128) as *const __m512i) };
        let chunk3 = unsafe { _mm512_loadu_si512(base_ptr.add(192) as *const __m512i) };

        let shuf0 = _mm512_shuffle_epi8(lookup_mask, _mm512_and_si512(chunk0, low_nibble_mask));
        let shuf1 = _mm512_shuffle_epi8(lookup_mask, _mm512_and_si512(chunk1, low_nibble_mask));
        let shuf2 = _mm512_shuffle_epi8(lookup_mask, _mm512_and_si512(chunk2, low_nibble_mask));
        let shuf3 = _mm512_shuffle_epi8(lookup_mask, _mm512_and_si512(chunk3, low_nibble_mask));

        let delim_k0 = _mm512_cmplt_epi8_mask(chunk0, non_printable_mask);
        let delim_k1 = _mm512_cmplt_epi8_mask(chunk1, non_printable_mask);
        let delim_k2 = _mm512_cmplt_epi8_mask(chunk2, non_printable_mask);
        let delim_k3 = _mm512_cmplt_epi8_mask(chunk3, non_printable_mask);

        let m0 = !_mm512_mask_cmpeq_epi8_mask(delim_k0, shuf0, _mm512_setzero_si512());
        let m1 = !_mm512_mask_cmpeq_epi8_mask(delim_k1, shuf1, _mm512_setzero_si512());
        let m2 = !_mm512_mask_cmpeq_epi8_mask(delim_k2, shuf2, _mm512_setzero_si512());
        let m3 = !_mm512_mask_cmpeq_epi8_mask(delim_k3, shuf3, _mm512_setzero_si512());

        process_mask_generic!(m0, idx, 0, write_ptr, end_write_ptr, bytes_ptr, fallback_ptr, buffer_len, u64);
        process_mask_generic!(m1, idx, 64, write_ptr, end_write_ptr, bytes_ptr, fallback_ptr, buffer_len, u64);
        process_mask_generic!(m2, idx, 128, write_ptr, end_write_ptr, bytes_ptr, fallback_ptr, buffer_len, u64);
        process_mask_generic!(m3, idx, 192, write_ptr, end_write_ptr, bytes_ptr, fallback_ptr, buffer_len, u64);

        idx += 256;
    }

    consume_tail(bytes_ptr, len, &mut idx, &mut write_ptr, end_write_ptr, fallback_ptr, buffer_len)
}
