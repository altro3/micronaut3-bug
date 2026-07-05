use crate::tokenizer::trainer::flat_corpus::FlatCorpus;

pub struct ThreadDeltaWorker;

impl ThreadDeltaWorker {
    #[inline(always)]
    pub fn merge_at_node<F>(corpus: &mut FlatCorpus, left_node_idx: i32, weight: i64, new_id: u32, mut on_pair_change: F) -> bool
    where
        F: FnMut((u32, u32), i64),
    {
        let l_idx = left_node_idx as usize;
        let left_node = corpus.nodes[l_idx];

        if left_node.next == -1 {
            return false;
        }
        let r_idx = left_node.next as usize;
        let right_node = corpus.nodes[r_idx];

        let target_pair = (left_node.id, right_node.id);
        let far_left_idx = left_node.prev;
        let far_right_idx = right_node.next;

        on_pair_change(target_pair, -weight);

        if far_left_idx != -1 {
            let fl = corpus.nodes[far_left_idx as usize];
            on_pair_change((fl.id, left_node.id), -weight);
            on_pair_change((fl.id, new_id), weight);
        }

        if far_right_idx != -1 {
            let fr = corpus.nodes[far_right_idx as usize];
            on_pair_change((right_node.id, fr.id), -weight);
            on_pair_change((new_id, fr.id), weight);
        }

        corpus.nodes[l_idx].id = new_id;
        corpus.nodes[l_idx].next = far_right_idx;

        if far_right_idx != -1 {
            corpus.nodes[far_right_idx as usize].prev = left_node_idx;
        }

        corpus.nodes[r_idx].prev = -1;
        corpus.nodes[r_idx].next = -1;

        true
    }
}
