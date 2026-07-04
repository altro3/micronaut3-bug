use std::cmp::Ordering;

pub type TokenId = u32;
pub type BpeRank = u32;

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct FlatBpeNode {
    pub id: u32,
    pub next: u16,
    pub prev: u16,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct MergePair {
    pub rank: u32,
    pub left_idx: u16,
    pub generation: u16,
}

impl Ord for MergePair {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank.cmp(&self.rank).then_with(|| self.left_idx.cmp(&other.left_idx))
    }
}

impl PartialOrd for MergePair {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
