use std::cmp::Ordering;

/// Элемент бинарной кучи для min-heap трекинга минимального ранга.
/// Размер структуры — 8 байт (u32 ранг + u16 позиция + u16 валидационный ID).
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct MergePair {
    pub rank: u32,
    pub left_idx: u16,
    pub generation: u16, // Для ленивой инвалидации устаревших пар
}

impl Ord for MergePair {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Минимальный ранг должен быть на вершине кучи (инвертируем)
        other.rank.cmp(&self.rank).then_with(|| self.left_idx.cmp(&other.left_idx))
    }
}

impl PartialOrd for MergePair {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
