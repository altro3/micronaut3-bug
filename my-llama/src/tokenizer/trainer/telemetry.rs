use crate::tokenizer::trainer::job::UltraJob;
use crate::tokenizer::trainer::position_index::PositionIndex;
use dary_heap::OctonaryHeap;
use std::time::Instant;

pub struct BpeTelemetry;

impl BpeTelemetry {
    pub fn log_progress(
        merges_done: usize,
        num_merges: usize,
        current_id: u32,
        job: &UltraJob,
        vocab_bytes_flat: &[u8],
        vocab_offsets_flat: &[u64],
        index: &PositionIndex,
        heap: &OctonaryHeap<UltraJob>,
        loop_start: Instant,
    ) {
        if merges_done.is_multiple_of(1000) || merges_done < 20 {
            let packed_a = vocab_offsets_flat[job.pair.0 as usize];
            let packed_b = vocab_offsets_flat[job.pair.1 as usize];

            let offset_a = (packed_a >> 32) as usize;
            let len_a = (packed_a & 0xFFFFFFFF) as usize;

            let offset_b = (packed_b >> 32) as usize;
            let len_b = (packed_b & 0xFFFFFFFF) as usize;

            let mut real_bytes = Vec::with_capacity(len_a + len_b);
            real_bytes.extend_from_slice(&vocab_bytes_flat[offset_a..offset_a + len_a]);
            real_bytes.extend_from_slice(&vocab_bytes_flat[offset_b..offset_b + len_b]);

            let clean_text = String::from_utf8_lossy(&real_bytes).into_owned();
            let pct = (merges_done as f64 / num_merges as f64) * 100.0;

            let elapsed_loop = loop_start.elapsed().as_secs_f64();
            let speed_merges = if elapsed_loop > 0.0 { merges_done as f64 / elapsed_loop } else { 0.0 };

            println!(
                "[BPE LOOP] Итерация #{:<5} ({:04.1}%) | Сила: {:<10} | Токен ID: {:<5} | Слияний/сек: {:.0} | Текст: '{}'",
                merges_done,
                pct,
                job.count,
                current_id,
                speed_merges,
                clean_text.escape_debug()
            );

            if merges_done.is_multiple_of(1000) && merges_done > 0 {
                println!(
                    "  [МОНИТОРИНГ ОЗУ] Активных пар: {} | Записей в индексе: {} | Размер кучи: {}",
                    index.pair_counts.len(),
                    index.pair_to_words.len(),
                    heap.len()
                );
            }
        }
    }
}
