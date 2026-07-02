#[macro_export]
macro_rules! process_mask_generic {
    ($mask_expr:expr, $idx:expr, $offset:expr, $write_ptr:ident, $end_write_ptr:ident, $bytes_ptr:ident, $fallback_ptr:ident, $buffer_len:ident, $t:ty) => {{
        let mut m = $mask_expr as $t;
        let chunk_offset = $idx + $offset;
        while m != 0 {
            let tz = m.trailing_zeros() as usize;
            if $write_ptr >= $end_write_ptr {
                return $buffer_len;
            }

            let byte_val = unsafe { *$bytes_ptr.add(chunk_offset + tz) };
            unsafe {
                *$write_ptr = *$fallback_ptr.add(byte_val as usize);
            }
            $write_ptr = unsafe { $write_ptr.add(1) };

            m &= m - 1;
        }
    }};
}
