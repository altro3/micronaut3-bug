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

        let mut input = regex_automata::Input::new(bytes);
        let mut last_end = 0usize;

        unsafe {
            let offsets_ptr = offsets_buffer.as_mut_ptr();
            let len_ptr = len_buffer.as_mut_ptr();

            while last_end < len && token_count < max_tokens {
                input.set_span(last_end..len);

                match self.dfa.try_search_fwd(&input) {
                    Ok(Some(half_match)) => {
                        let start = last_end;
                        let end = half_match.offset();

                        if end == start {
                            last_end += 1;
                            continue;
                        }

                        let match_len = end - start;
                        *offsets_ptr.add(token_count) = start as u32;
                        *len_ptr.add(token_count) = match_len as u32;
                        token_count += 1;

                        last_end = end;
                    }
                    _ => {
                        break;
                    }
                }
            }

            if last_end < len && token_count < max_tokens {
                *offsets_ptr.add(token_count) = last_end as u32;
                *len_ptr.add(token_count) = (len - last_end) as u32;
                token_count += 1;
            }
        }

        token_count
    }
}
