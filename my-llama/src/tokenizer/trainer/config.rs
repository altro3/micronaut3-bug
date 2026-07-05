// src/tokenizer/trainer/config.rs

pub struct TrainerConfig {
    pub batch_size: usize,
    pub initial_table_size: usize,
    pub io_buffer_size: usize,
    pub start_token_id: u32,
    pub num_threads: usize,
    pub local_map_capacity: usize,       // Размер локальной мапы сбора слов
    pub delta_map_capacity: usize,       // Размер мапы дельт частот пар
    pub position_buffer_capacity: usize,  // Начальный буфер очагов мутаций
    pub index_rebuild_interval: usize,    // Частота дефрагментации индекса
}

impl Default for TrainerConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            batch_size: 256,
            initial_table_size: 524288,
            io_buffer_size: 4 * 1024 * 1024,
            start_token_id: 256,
            num_threads: 16,
            local_map_capacity: 16384,
            delta_map_capacity: 4096,
            position_buffer_capacity: 65536,
            index_rebuild_interval: 16,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PositionMatch {
    pub pos: usize,
    pub id1: u32,
    pub id2: u32,
    pub new_id: u32,
}
