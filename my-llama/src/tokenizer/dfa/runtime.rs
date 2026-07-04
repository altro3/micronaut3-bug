use super::table::DfaTransitionTable;

pub struct FlatDfaRuntime {
    table: DfaTransitionTable,
}

impl FlatDfaRuntime {
    pub fn from_binary_dump(num_states: usize, trans_bytes: &[u8], accept_bytes: &[u8]) -> Self {
        let mut table = DfaTransitionTable::new(num_states);
        unsafe {
            let dst_ptr = table.ptr_mut();
            std::ptr::copy_nonoverlapping(trans_bytes.as_ptr(), dst_ptr as *mut u8, trans_bytes.len());

            for state in 0..num_states {
                for byte in 0..=255 {
                    let idx = (state << 8) | byte;
                    let target_state = *dst_ptr.add(idx);
                    if target_state != 0xFFFF && *accept_bytes.get_unchecked(target_state as usize) != 0 {
                        *dst_ptr.add(idx) = target_state | 0x8000;
                    }
                }
            }
        }
        Self { table }
    }

    #[inline(always)]
    pub fn split_streaming(&self, bytes: &[u8], offsets_buffer: &mut [u32], len_buffer: &mut [u32]) -> usize {
        let len = bytes.len();
        if len == 0 {
            return 0;
        }

        let mut token_count = 0;
        let max_tokens = offsets_buffer.len();
        let bytes_ptr = bytes.as_ptr();

        let mut start_idx = 0usize;
        let mut curr_idx = 0usize;
        let mut state = 0u16;
        let mut last_accept_idx = None;

        unsafe {
            while curr_idx < len && token_count < max_tokens {
                let byte = *bytes_ptr.add(curr_idx);
                let raw_next = self.table.get_next_state(state, byte);

                if raw_next == 0xFFFF {
                    if let Some(accept_idx) = last_accept_idx {
                        let token_len = accept_idx - start_idx;
                        *offsets_buffer.get_unchecked_mut(token_count) = start_idx as u32;
                        *len_buffer.get_unchecked_mut(token_count) = token_len as u32;
                        token_count += 1;

                        start_idx = accept_idx;
                        curr_idx = start_idx;
                    } else {
                        *offsets_buffer.get_unchecked_mut(token_count) = start_idx as u32;
                        *len_buffer.get_unchecked_mut(token_count) = 1;
                        token_count += 1;

                        start_idx += 1;
                        curr_idx = start_idx;
                    }
                    state = 0;
                    last_accept_idx = None;
                    continue;
                }

                state = raw_next & 0x7FFF;
                curr_idx += 1;

                if (raw_next & 0x8000) != 0 {
                    last_accept_idx = Some(curr_idx);
                }
            }

            if start_idx < len && token_count < max_tokens {
                let final_len = last_accept_idx.unwrap_or(len) - start_idx;
                *offsets_buffer.get_unchecked_mut(token_count) = start_idx as u32;
                *len_buffer.get_unchecked_mut(token_count) = if final_len > 0 { final_len as u32 } else { (len - start_idx) as u32 };
                token_count += 1;
            }
        }

        token_count
    }
}
