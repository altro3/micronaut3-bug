use std::cmp::Ordering;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct BpePair {
    pub rank: u32,
    pub index: usize,
}

impl Ord for BpePair {
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank.cmp(&self.rank)
            .then(other.index.cmp(&self.index))
    }
}

impl PartialOrd for BpePair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
