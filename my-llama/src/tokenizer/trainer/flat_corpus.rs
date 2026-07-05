use fxhash::FxHasher;
use std::hash::BuildHasherDefault;

type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Copy, Clone, Debug)]
pub struct TokenNode {
    pub id: u32,
    pub prev: i32,
    pub next: i32,
}

#[derive(Clone, Debug)]
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
    pub pair_heads: FxHashMap<(u32, u32), i32>,
    pub links: Vec<i32>,
    pub word_indices: Vec<usize>,
    pub node_indices: Vec<i32>,
}

impl FlatCorpus {
    pub fn build(unique_words: FxHashMap<Vec<u8>, usize>) -> (Self, ProPairIndex) {
        let mut total_tokens = 0;
        for (bytes, _) in &unique_words {
            total_tokens += bytes.len();
        }

        let mut nodes = Vec::with_capacity(total_tokens);
        let mut words = Vec::with_capacity(unique_words.len());
        let mut pair_counts = FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());

        let mut temp_pair_map: FxHashMap<(u32, u32), Vec<(usize, i32)>> =
            FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());

        for (bytes, count) in unique_words {
            if bytes.is_empty() {
                continue;
            }
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
            }

            words.push(WordSlice {
                head: start_node_idx,
                weight,
            });

            for i in 0..(len - 1) {
                let n_idx = start_node_idx + i as i32;
                let pair = (nodes[n_idx as usize].id, nodes[(n_idx + 1) as usize].id);
                *pair_counts.entry(pair).or_insert(0) += weight;
                temp_pair_map.entry(pair).or_insert_with(Vec::new).push((word_idx, n_idx));
            }
        }

        let total_pairs_count: usize = temp_pair_map.values().map(|v| v.len()).sum();
        let mut pair_heads = FxHashMap::with_capacity_and_hasher(temp_pair_map.len(), Default::default());
        let mut links = vec![-1; total_pairs_count];
        let mut word_indices = Vec::with_capacity(total_pairs_count);
        let mut node_indices = Vec::with_capacity(total_pairs_count);

        let mut current_link_idx = 0;
        for (pair, positions) in temp_pair_map {
            let mut head = -1;
            for (w_idx, n_idx) in positions {
                word_indices.push(w_idx);
                node_indices.push(n_idx);
                links[current_link_idx] = head;
                head = current_link_idx as i32;
                current_link_idx += 1;
            }
            pair_heads.insert(pair, head);
        }

        (
            Self { nodes, words },
            ProPairIndex {
                pair_counts,
                pair_heads,
                links,
                word_indices,
                node_indices,
            },
        )
    }
}
