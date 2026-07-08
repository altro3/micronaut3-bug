use dary_heap::OctonaryHeap;
use fxhash::FxHashMap;
use std::io::{Error, ErrorKind};
use std::time::Instant;

use crate::tokenizer::trainer::aggregator::CorpusAggregator;
use crate::tokenizer::trainer::bpe_loop::BpeLoopRunner;
use crate::tokenizer::trainer::config::TrainerConfig;
use crate::tokenizer::trainer::exporter::VocabularyExporter;
use crate::tokenizer::trainer::job::UltraJob;
use crate::tokenizer::trainer::position_index::PositionIndex;
use crate::tokenizer::trainer::utils::TrainerUtils;

pub struct BpeTrainer {
    config: TrainerConfig,
}

impl BpeTrainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    pub fn train(&self, text_content: &str, output_json_path: &str) -> std::io::Result<()> {
        if self.config.vocab_size < 256 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "Размер словаря (vocab_size) не может быть меньше 256! У вас указано: {}. \
                Укажите как минимум 256 + количество планируемых слияний.",
                    self.config.vocab_size
                ),
            ));
        }

        let global_start = Instant::now();

        let raw_words = self.aggregate_corpus(text_content);

        let (mut words, mut index) = PositionIndex::build(raw_words);

        let (mut vocab_bytes_flat, mut vocab_offsets_flat) = self.init_base_alphabet();
        let mut heap = self.init_priority_queue(&index.pair_counts);

        let loop_runner = BpeLoopRunner::new(&self.config);
        let raw_merges = loop_runner.run(&mut words, &mut index, &mut heap, &mut vocab_bytes_flat, &mut vocab_offsets_flat);
        self.finalize_and_export(&vocab_bytes_flat, &vocab_offsets_flat, raw_merges, output_json_path, global_start)?;

        Ok(())
    }

    fn aggregate_corpus(&self, text_content: &str) -> FxHashMap<Vec<u8>, usize> {
        let aggregator = CorpusAggregator::new(&self.config.regex, self.config.initial_table_size);
        aggregator.collect_unique_words(text_content.as_bytes(), self.config.num_threads, self.config.local_map_capacity)
    }

    fn init_base_alphabet(&self) -> (Vec<u8>, Vec<u64>) {
        let mut vocab_bytes_flat = Vec::with_capacity(self.config.vocab_size * 8);
        let mut vocab_offsets_flat = vec![0u64; self.config.vocab_size];

        for (b, offset_slot) in vocab_offsets_flat.iter_mut().enumerate().take(256) {
            let offset = vocab_bytes_flat.len() as u64;
            vocab_bytes_flat.push(b as u8);
            *offset_slot = (offset << 32) | 1u64;
        }
        (vocab_bytes_flat, vocab_offsets_flat)
    }

    fn init_priority_queue(&self, pair_counts: &FxHashMap<(u32, u32), i64>) -> OctonaryHeap<UltraJob> {
        println!("[ТРЕНЕР] Заполнение приоритетной очереди OctonaryHeap...");
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (&pair, &count) in pair_counts {
            if count > 0 {
                heap.push(UltraJob { count, pair });
            }
        }
        println!("  └── Очередь готова. Записей в куче: {}", heap.len());
        heap
    }

    fn finalize_and_export(
        &self,
        vocab_bytes_flat: &[u8],
        vocab_offsets_flat: &[u64],
        raw_merges: Vec<(u32, u32)>,
        output_json_path: &str,
        global_start: Instant,
    ) -> std::io::Result<()> {
        println!("[ФИНАЛИЗАЦИЯ] Конвертация токенов в формат Qwen JSON напрямую из Flat Buffer...");
        let mut vocab_json_output = FxHashMap::with_capacity_and_hasher(vocab_offsets_flat.len(), Default::default());

        for (b, &packed) in vocab_offsets_flat.iter().enumerate().take(256) {
            let offset = (packed >> 32) as usize;
            let length = (packed & 0xFFFFFFFF) as usize;
            let slice = &vocab_bytes_flat[offset..offset + length];

            let qwen_str = TrainerUtils::bytes_to_qwen_string(slice);
            vocab_json_output.insert(qwen_str, b as u32);
        }

        let mut merges: Vec<[String; 2]> = Vec::with_capacity(raw_merges.len());
        for (idx, (p0, p1)) in raw_merges.into_iter().enumerate() {
            let packed_a = vocab_offsets_flat[p0 as usize];
            let offset_a = (packed_a >> 32) as usize;
            let len_a = (packed_a & 0xFFFFFFFF) as usize;
            let str_a = TrainerUtils::bytes_to_qwen_string(&vocab_bytes_flat[offset_a..offset_a + len_a]);

            let packed_b = vocab_offsets_flat[p1 as usize];
            let offset_b = (packed_b >> 32) as usize;
            let len_b = (packed_b & 0xFFFFFFFF) as usize;
            let str_b = TrainerUtils::bytes_to_qwen_string(&vocab_bytes_flat[offset_b..offset_b + len_b]);

            merges.push([str_a, str_b]);

            let token_id = self.config.start_token_id + idx as u32;
            let packed_m = vocab_offsets_flat[token_id as usize];
            let offset_m = (packed_m >> 32) as usize;
            let len_m = (packed_m & 0xFFFFFFFF) as usize;
            let str_merged = TrainerUtils::bytes_to_qwen_string(&vocab_bytes_flat[offset_m..offset_m + len_m]);

            vocab_json_output.insert(str_merged, token_id);
        }

        let total_merges_count = merges.len() as u32;
        println!("[ЭКСПОРТ] Запись совместимого JSON на диск: {}", output_json_path);

        let mut std_vocab = FxHashMap::with_capacity_and_hasher(vocab_json_output.len(), Default::default());
        for (k, v) in vocab_json_output {
            std_vocab.insert(k, v);
        }

        VocabularyExporter::export_qwen_json(
            output_json_path,
            &self.config,
            &self.config.regex,
            std_vocab,
            merges,
            self.config.start_token_id + total_merges_count,
        )?;

        println!("[ГОТОВО] Общее время работы всего канонического пайплайна: {:?}", global_start.elapsed());
        Ok(())
    }
}
