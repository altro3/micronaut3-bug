use crate::tokenizer::simd::utils::consume_tail;
use std::arch::x86_64::*;

#[repr(align(32))]
struct AlignedLookup {
    data: [i8; 32],
}

static LOOKUP_MASK: AlignedLookup = AlignedLookup {
    data: [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    ],
};

#[target_feature(enable = "avx2,bmi2")]
pub unsafe fn split(text: &str, ids_buffer: &mut [u32], byte_fallback: &[u32; 256]) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut idx = 0;

    let buffer_len = ids_buffer.len();
    let buf_start_ptr = ids_buffer.as_mut_ptr();
    let mut write_ptr = buf_start_ptr;
    let end_write_ptr = unsafe { write_ptr.add(buffer_len) };

    let fallback_ptr = byte_fallback.as_ptr() as *const i32;

    let lookup_mask = unsafe { _mm256_load_si256((&LOOKUP_MASK.data as *const [i8; 32]) as *const __m256i) };
    let low_nibble_mask = _mm256_set1_epi8(0x0F);
    let non_printable_mask = _mm256_set1_epi8(33);

    while idx + 32 <= len {
        let base_ptr = unsafe { bytes.as_ptr().add(idx) };

        let chunk = unsafe { _mm256_loadu_si256(base_ptr as *const __m256i) };

        let delim = _mm256_and_si256(
            _mm256_shuffle_epi8(lookup_mask, _mm256_and_si256(chunk, low_nibble_mask)),
            _mm256_cmpgt_epi8(non_printable_mask, chunk),
        );
        let delim_mask = !_mm256_movemask_epi8(delim);

        let (ids0, m0, ids1, m1, ids2, m2, ids3, m3) = unsafe {
            let text_vec0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(base_ptr as *const __m128i));
            let ids0 = _mm256_i32gather_epi32(fallback_ptr, text_vec0, 4);

            let text_vec1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(base_ptr.add(8) as *const __m128i));
            let ids1 = _mm256_i32gather_epi32(fallback_ptr, text_vec1, 4);

            let text_vec2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(base_ptr.add(16) as *const __m128i));
            let ids2 = _mm256_i32gather_epi32(fallback_ptr, text_vec2, 4);

            let text_vec3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(base_ptr.add(24) as *const __m128i));
            let ids3 = _mm256_i32gather_epi32(fallback_ptr, text_vec3, 4);

            (
                ids0,
                delim_mask & 0xFF,
                ids1,
                (delim_mask >> 8) & 0xFF,
                ids2,
                (delim_mask >> 16) & 0xFF,
                ids3,
                (delim_mask >> 24) & 0xFF,
            )
        };

        unsafe {
            _mm256_maskstore_epi32(write_ptr as *mut i32, _mm256_set1_epi32(m0), ids0);
            write_ptr = write_ptr.add(m0.count_ones() as usize);

            _mm256_maskstore_epi32(write_ptr as *mut i32, _mm256_set1_epi32(m1), ids1);
            write_ptr = write_ptr.add(m1.count_ones() as usize);

            _mm256_maskstore_epi32(write_ptr as *mut i32, _mm256_set1_epi32(m2), ids2);
            write_ptr = write_ptr.add(m2.count_ones() as usize);

            _mm256_maskstore_epi32(write_ptr as *mut i32, _mm256_set1_epi32(m3), ids3);
            write_ptr = write_ptr.add(m3.count_ones() as usize);
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
