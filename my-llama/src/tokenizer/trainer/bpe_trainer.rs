use dary_heap::OctonaryHeap;
use fxhash::FxHasher;
use std::hash::BuildHasherDefault;
use std::io::{Error, ErrorKind};
use std::time::Instant;

type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

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
                Укажите как минимум 256 + количество планируемых слияний (например, 266).",
                    self.config.vocab_size
                ),
            ));
        }

        let global_start = Instant::now();

        let raw_words = self.aggregate_corpus(text_content);

        // Работаем по нашему стабильному координатному индексу
        let (mut words, mut index) = PositionIndex::build(raw_words);

        let mut id_to_bytes = self.init_base_alphabet();
        let mut heap = self.init_priority_queue(&index.pair_counts);

        let loop_runner = BpeLoopRunner::new(&self.config);
        let raw_merges = loop_runner.run(&mut words, &mut index, &mut heap, &mut id_to_bytes);

        self.finalize_and_export(id_to_bytes, raw_merges, output_json_path, global_start)?;

        Ok(())
    }

    fn aggregate_corpus(&self, text_content: &str) -> FxHashMap<Vec<u8>, usize> {
        let aggregator = CorpusAggregator::new(&self.config.regex, self.config.initial_table_size);
        let raw_words = aggregator.collect_unique_words(text_content.as_bytes(), self.config.num_threads, self.config.local_map_capacity);
        let mut unique_words = FxHashMap::with_capacity_and_hasher(raw_words.len(), Default::default());
        for (k, v) in raw_words {
            unique_words.insert(k, v);
        }
        unique_words
    }

    fn init_base_alphabet(&self) -> Vec<Vec<u8>> {
        let mut id_to_bytes: Vec<Vec<u8>> = (0..256).map(|b| vec![b as u8]).collect();
        if (self.config.start_token_id as usize) > id_to_bytes.len() {
            id_to_bytes.resize(self.config.start_token_id as usize, vec![]);
        }
        id_to_bytes.reserve(self.config.vocab_size);
        id_to_bytes
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
        id_to_bytes: Vec<Vec<u8>>,
        raw_merges: Vec<(u32, u32)>,
        output_json_path: &str,
        global_start: std::time::Instant,
    ) -> std::io::Result<()> {
        println!("[ФИНАЛИЗАЦИЯ] Конвертация токенов в формат Qwen JSON...");
        let mut vocab_json_output = FxHashMap::with_capacity_and_hasher(id_to_bytes.len(), Default::default());

        for b in 0..256 {
            let b_vec = vec![b as u8];
            let qwen_str = TrainerUtils::bytes_to_qwen_string(&b_vec);
            vocab_json_output.insert(qwen_str, b as u32);
        }

        let mut merges: Vec<[String; 2]> = Vec::with_capacity(raw_merges.len());
        for (idx, (p0, p1)) in raw_merges.into_iter().enumerate() {
            let str_a = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[p0 as usize]);
            let str_b = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[p1 as usize]);
            merges.push([str_a, str_b]);

            let token_id = self.config.start_token_id + idx as u32;
            let str_merged = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[token_id as usize]);
            vocab_json_output.insert(str_merged, token_id);
        }

        let mut std_vocab = std::collections::HashMap::with_capacity(vocab_json_output.len());
        for (k, v) in vocab_json_output {
            std_vocab.insert(k, v);
        }

        let total_merges_count = merges.len() as u32;
        println!("[ЭКСПОРТ] Запись совместимого JSON на disk: {}", output_json_path);
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
