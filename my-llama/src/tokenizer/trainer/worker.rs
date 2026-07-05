use crate::tokenizer::trainer::config::PositionMatch;
use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use std::collections::HashMap;

pub struct ThreadDeltaWorker;

impl ThreadDeltaWorker {
    pub fn process_chunk(p_chunk: &[PositionMatch], corpus: &FlatCorpus, local_delta: &mut HashMap<(u32, u32), i64>) {
        let tokens_ptr = corpus.tokens.as_ptr() as *mut u32;
        let next_ptr = corpus.next.as_ptr() as *mut i32;
        let prev_ptr = corpus.prev.as_ptr() as *mut i32;

        unsafe {
            let tokens = std::slice::from_raw_parts_mut(tokens_ptr, corpus.tokens.len());
            let next = std::slice::from_raw_parts_mut(next_ptr, corpus.next.len());
            let prev = std::slice::from_raw_parts_mut(prev_ptr, corpus.prev.len());

            for item in p_chunk {
                let pos = item.pos;
                if tokens[pos] != item.id1 || next[pos] == -1 || tokens[next[pos] as usize] != item.id2 {
                    continue;
                }

                let w_idx = corpus.find_word_index(pos);
                let weight = corpus.word_counts[w_idx] as i64;

                let next_node = next[pos] as usize;
                let right_neighbor = next[next_node];
                let left_neighbor = prev[pos];

                if left_neighbor != -1 {
                    let old_pair = (tokens[left_neighbor as usize], tokens[pos]);
                    *local_delta.entry(old_pair).or_insert(0) -= weight;
                }
                if right_neighbor != -1 {
                    let old_pair = (tokens[next_node], tokens[right_neighbor as usize]);
                    *local_delta.entry(old_pair).or_insert(0) -= weight;
                }

                tokens[pos] = item.new_id;
                next[pos] = right_neighbor;
                if right_neighbor != -1 {
                    prev[right_neighbor as usize] = pos as i32;
                }

                if left_neighbor != -1 {
                    let new_pair = (tokens[left_neighbor as usize], tokens[pos]);
                    *local_delta.entry(new_pair).or_insert(0) += weight;
                }
                if right_neighbor != -1 {
                    let new_pair = (tokens[pos], tokens[right_neighbor as usize]);
                    *local_delta.entry(new_pair).or_insert(0) += weight;
                }
            }
        }
    }
}
