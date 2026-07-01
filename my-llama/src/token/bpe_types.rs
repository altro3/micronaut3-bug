use std::cmp::Ordering;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct FlatToken {
    pub hash: u64,
    pub id: u32,
}

impl Ord for FlatToken {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hash.cmp(&other.hash)
    }
}

impl PartialOrd for FlatToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Copy, Clone)]
pub struct BpeChunk {
    pub start: usize,
    pub end: usize,
}
