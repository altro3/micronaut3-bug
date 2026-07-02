#[inline(always)]
pub fn consume_tail(
    bytes_ptr: *const u8,
    len: usize,
    idx: &mut usize,
    write_ptr: &mut *mut u32,
    end_write_ptr: *mut u32,
    fallback_ptr: *const u32,
    buffer_len: usize,
) -> usize {
    while *idx < len {
        let b = unsafe { *bytes_ptr.add(*idx) };
        let is_delim = if b <= 32 { ((1u64 << b) & 0x10000000600u64) != 0 } else { false };

        if !is_delim {
            if *write_ptr >= end_write_ptr {
                return buffer_len;
            }
            unsafe {
                **write_ptr = *fallback_ptr.add(b as usize);
            }
            *write_ptr = unsafe { (*write_ptr).add(1) };
        }
        *idx += 1;
    }
    unsafe { (*write_ptr).offset_from(bytes_ptr as *const u32) as usize }
}
