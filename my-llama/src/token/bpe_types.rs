use std::cmp::Ordering;

#[derive(Eq, PartialEq)]
pub struct BpePair {
    pub rank: u32,
    pub left_idx: usize,
}

impl Ord for BpePair {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank.cmp(&self.rank).then_with(|| self.left_idx.cmp(&other.left_idx))
    }
}

impl PartialOrd for BpePair {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BpeValue {
    pub rank: u32,
    pub id: u32,
}
