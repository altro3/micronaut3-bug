use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;

pub struct FlatDfaRuntime {
    dfa: DFA<Vec<u32>>,
}

impl FlatDfaRuntime {
    pub fn from_binary_dump(_num_states: usize, trans_bytes: &[u8], _accept_bytes: &[u8]) -> Self {
        let (borrowed_dfa, _) = DFA::from_bytes(trans_bytes).expect("Критический сбой десериализации матрицы DFA!");
        let dfa = borrowed_dfa.to_owned();

        Self { dfa }
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

        let mut state = self.dfa.start_state_forward(&regex_automata::Input::new("")).unwrap();
        let mut last_accept_idx = None;

        unsafe {
            while curr_idx < len && token_count < max_tokens {
                let byte = *bytes_ptr.add(curr_idx);
                let next_state = self.dfa.next_state(state, byte);

                if self.dfa.is_dead_state(next_state) || self.dfa.is_quit_state(next_state) {
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
                    state = self.dfa.start_state_forward(&regex_automata::Input::new("")).unwrap();
                    last_accept_idx = None;
                    continue;
                }

                state = next_state;
                curr_idx += 1;

                if self.dfa.is_match_state(state) {
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
