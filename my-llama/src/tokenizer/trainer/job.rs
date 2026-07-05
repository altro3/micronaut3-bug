use core::cmp::Ordering;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct UltraJob {
    pub count: i64,
    pub pair: (u32, u32),
}

impl Ord for UltraJob {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count).then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for UltraJob {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
