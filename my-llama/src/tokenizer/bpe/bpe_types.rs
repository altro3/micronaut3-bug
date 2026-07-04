pub type TokenId = u32;
pub type BpeRank = u32;

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct FlatBpeNode {
    pub id: u32,
    pub next: u16,
    pub prev: u16,
}
