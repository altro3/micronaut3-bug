use super::table::DfaTransitionTable;

pub struct FlatDfaRuntime {
    table: DfaTransitionTable,
    accept_states: Vec<bool>,
}

impl FlatDfaRuntime {
    pub fn from_binary_dump(num_states: usize, trans_bytes: &[u8], accept_bytes: &[u8]) -> Self {
        let mut table = DfaTransitionTable::new(num_states);

        unsafe {
            let dst_ptr = table.ptr_mut();
            std::ptr::copy_nonoverlapping(trans_bytes.as_ptr(), dst_ptr as *mut u8, trans_bytes.len());
        }

        let mut accept_states = vec![false; num_states];
        for (i, &b) in accept_bytes.iter().enumerate() {
            if i < num_states {
                accept_states[i] = b != 0;
            }
        }

        Self { table, accept_states }
    }

    #[inline(always)]
    pub fn split_streaming(&self, bytes: &[u8], offsets_buffer: &mut [u32], len_buffer: &mut [u32]) -> usize {
        let len = bytes.len();
        if len == 0 {
            return 0;
        }

        let mut token_count = 0;
        let max_tokens = offsets_buffer.len();

        let accept_ptr = self.accept_states.as_ptr();
        let bytes_ptr = bytes.as_ptr();
        let mut start_idx = 0usize;

        while start_idx < len && token_count < max_tokens {
            let mut state = 0u16;
            let mut curr_idx = start_idx;
            let mut last_accept_idx = start_idx;

            unsafe {
                while curr_idx < len {
                    let byte = *bytes_ptr.add(curr_idx);
                    let next_state = self.table.get_next_state(state, byte);

                    if next_state == 0xFFFF {
                        break;
                    }

                    state = next_state;
                    curr_idx += 1;

                    if *accept_ptr.add(state as usize) {
                        last_accept_idx = curr_idx;
                    }
                }

                if last_accept_idx > start_idx {
                    *offsets_buffer.get_unchecked_mut(token_count) = start_idx as u32;
                    *len_buffer.get_unchecked_mut(token_count) = (last_accept_idx - start_idx) as u32;
                    token_count += 1;
                    start_idx = last_accept_idx;
                } else {
                    *offsets_buffer.get_unchecked_mut(token_count) = start_idx as u32;
                    *len_buffer.get_unchecked_mut(token_count) = 1;
                    token_count += 1;
                    start_idx += 1;
                }
            }
        }

        token_count
    }
}
