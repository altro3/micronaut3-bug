pub struct TrainerConfig {
    pub vocab_size: usize,
    pub batch_size: usize,
    pub initial_table_size: usize,
    pub io_buffer_size: usize,
    pub start_token_id: u32,
    pub num_threads: usize,
    pub local_map_capacity: usize,
    pub delta_map_capacity: usize,
    pub position_buffer_capacity: usize,
    pub index_rebuild_interval: usize,
    pub max_token_length: usize,
    pub regex: String,
}

impl Default for TrainerConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            batch_size: 256,
            initial_table_size: 524288,
            io_buffer_size: 4 * 1024 * 1024,
            start_token_id: 256,
            num_threads: 16,
            local_map_capacity: 16384,
            delta_map_capacity: 4096,
            position_buffer_capacity: 65536,
            index_rebuild_interval: 16,
            max_token_length: 16,
            regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+|(?:\s)[\r\n]*|\s+[\r\n]*|[\r\n]+".to_string(),
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

#[derive(Copy, Clone, Debug)]
pub struct MergeCommand {
    pub pair: (u32, u32),
    pub new_id: u32,
}
