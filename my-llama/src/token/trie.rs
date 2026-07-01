#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TrieNode {
    pub token_id: u32,
    pub children_base: u32,
}

pub struct FlatTrie {
    pub nodes: Vec<TrieNode>,
}

impl FlatTrie {
    pub fn new(nodes: Vec<TrieNode>) -> Self {
        Self { nodes }
    }

    #[inline(always)]
    pub fn transition(&self, current_node_idx: usize, byte: u8) -> Option<usize> {
        let node = unsafe { *self.nodes.get_unchecked(current_node_idx) };
        if node.children_base == u32::MAX {
            return None;
        }

        let next_node_idx = node.children_base as usize + byte as usize;
        if next_node_idx >= self.nodes.len() {
            return None;
        }

        let next_node = unsafe { *self.nodes.get_unchecked(next_node_idx) };

        if next_node.children_base == 0 && next_node.token_id == u32::MAX {
            return None;
        }

        Some(next_node_idx)
    }

    #[inline(always)]
    pub fn get_token_id(&self, node_idx: usize) -> u32 {
        unsafe { self.nodes.get_unchecked(node_idx).token_id }
    }
}
