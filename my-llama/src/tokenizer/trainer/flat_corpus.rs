use std::hash::BuildHasherDefault;
use fxhash::FxHasher;

type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Copy, Clone, Debug)]
pub struct TokenNode {
    pub id: u32,
    pub prev: i32,
    pub next: i32,
}

pub struct WordSlice {
    pub head: i32,
    pub weight: i64,
}

pub struct FlatCorpus {
    pub nodes: Vec<TokenNode>,
    pub words: Vec<WordSlice>,
}

pub struct ProPairIndex {
    pub pair_counts: FxHashMap<(u32, u32), i64>,
    pub pair_slices: FxHashMap<(u32, u32), (usize, usize)>,
    pub positions: Vec<i32>,
    pub node_to_word: Vec<usize>,
}

impl FlatCorpus {
    pub fn build(unique_words: FxHashMap<Vec<u8>, usize>) -> (Self, ProPairIndex) {
        let mut total_tokens = 0;
        for (bytes, _) in &unique_words {
            total_tokens += bytes.len();
        }

        let mut nodes = Vec::with_capacity(total_tokens);
        let mut words = Vec::with_capacity(unique_words.len());
        let mut node_to_word = Vec::with_capacity(total_tokens);
        let mut pair_counts = FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());

        let mut temp_pair_map: FxHashMap<(u32, u32), Vec<i32>> =
            FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());

        for (bytes, count) in unique_words {
            if bytes.is_empty() { continue; }
            let weight = count as i64;
            let word_idx = words.len();
            let start_node_idx = nodes.len() as i32;

            let len = bytes.len();
            for i in 0..len {
                nodes.push(TokenNode {
                    id: bytes[i] as u32,
                    prev: if i == 0 { -1 } else { start_node_idx + i as i32 - 1 },
                    next: if i == len - 1 { -1 } else { start_node_idx + i as i32 + 1 },
                });
                node_to_word.push(word_idx);
            }

            words.push(WordSlice { head: start_node_idx, weight });

            for i in 0..(len - 1) {
                let n_idx = start_node_idx + i as i32;
                let pair = (nodes[n_idx as usize].id, nodes[(n_idx + 1) as usize].id);
                *pair_counts.entry(pair).or_insert(0) += weight;
                temp_pair_map.entry(pair).or_insert_with(Vec::new).push(n_idx);
            }
        }

        let total_positions: usize = temp_pair_map.values().map(|v| v.len()).sum();
        let mut pair_slices = FxHashMap::with_capacity_and_hasher(temp_pair_map.len(), Default::default());
        let mut positions = Vec::with_capacity(total_positions);

        for (pair, node_indices) in temp_pair_map {
            let start = positions.len();
            let count = node_indices.len();
            positions.extend_from_slice(&node_indices);
            pair_slices.insert(pair, (start, count));
        }

        (
            Self { nodes, words },
            ProPairIndex { pair_counts, pair_slices, positions, node_to_word }
        )
    }
}
