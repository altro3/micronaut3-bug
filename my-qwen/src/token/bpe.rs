use base64::{prelude::BASE64_STANDARD, Engine};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
// Импортируем движок Base64

pub struct BpeTokenizer {
    ranks: HashMap<(u8, u8), u32>,
    encoder: HashMap<Vec<u8>, u32>,
    decoder: HashMap<u32, Vec<u8>>,
    pub eos_token_id: u32,
}

impl BpeTokenizer {
    pub fn new_micro() -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut ranks = HashMap::new();

        // 1. Базовые байты (0..=255) строго под своими ID
        for b in 0..=255 {
            let bytes = vec![b];
            let id = b as u32;
            encoder.insert(bytes.clone(), id);
            decoder.insert(id, bytes);
        }

        // 2. Выделяем ID 256 под [EOS]
        let eos_bytes = b"[EOS]".to_vec();
        let eos_id = 256;
        encoder.insert(eos_bytes.clone(), eos_id);
        decoder.insert(eos_id, eos_bytes);

        // 3. Кастомные BPE-ранги (начинаются с ID 257)
        let mut current_id = 257;
        let mut rank_counter = 0;
        let mut add_bpe_merge = |b1: u8, b2: u8| {
            ranks.insert((b1, b2), rank_counter);
            rank_counter += 1;

            let merged_bytes = vec![b1, b2];
            if !encoder.contains_key(&merged_bytes) {
                encoder.insert(merged_bytes.clone(), current_id);
                decoder.insert(current_id, merged_bytes);
                current_id += 1;
            }
        };

        // Склеиваем байты кириллицы на уровне словаря
        add_bpe_merge(208, 191); // 'п'
        add_bpe_merge(209, 128); // 'р'

        BpeTokenizer {
            ranks,
            encoder,
            decoder,
            eos_token_id: eos_id,
        }
    }

    /// Промышленный конструктор: загружает реальный словарь Qwen из .tiktoken файла
    pub fn from_file(file_path: &str) -> std::io::Result<Self> {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut ranks = HashMap::new();

        // Открываем файл словаря с буферизацией для максимальной скорости чтения
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut max_id = 0;

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            // Строка имеет формат: "Base64_Строка ID"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }

            // 1. Декодируем Base64 в сырые байты токена
            let token_bytes = BASE64_STANDARD
                .decode(parts[0])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            // 2. Парсим ID токена
            let id = parts[1]
                .parse::<u32>()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            if id > max_id {
                max_id = id;
            }

            // 3. Заполняем основные мапы кодирования/декодирования
            encoder.insert(token_bytes.clone(), id);
            decoder.insert(id, token_bytes.clone());

            // 4. Если токен состоит ровно из двух байт, регистрируем его в таблице BPE-рангов!
            // Именно так восстанавливаются правила слияния из файла tiktoken
            if token_bytes.len() == 2 {
                let pair = (token_bytes[0], token_bytes[1]);
                ranks.insert(pair, id); // Ранг в Tiktoken равен самому ID токена
            }
        }

        // В оригинальном Qwen [EOS] имеет конкретный ID.
        // Если его нет в текстовом файле, выделим ему следующий свободный ID
        let eos_id = max_id + 1;
        let eos_bytes = b"<|endoftext|>".to_vec(); // Каноничный маркер Qwen/GPT
        encoder.insert(eos_bytes.clone(), eos_id);
        decoder.insert(eos_id, eos_bytes);

        println!(
            "Успешно загружен словарь Tiktoken! Всего токенов: {}",
            encoder.len()
        );

        Ok(BpeTokenizer {
            ranks,
            encoder,
            decoder,
            eos_token_id: eos_id,
        })
    }

    fn bpe_merge(&self, word_bytes: Vec<u8>) -> Vec<Vec<u8>> {
        // Инициализируем массив частей: изначально каждый байт — это отдельный вектор из 1 элемента
        let mut parts: Vec<Vec<u8>> = word_bytes.iter().map(|&b| vec![b]).collect();

        loop {
            if parts.len() < 2 {
                break;
            }

            let mut best_pair = None;
            let mut min_rank = u32::MAX;

            for i in 0..parts.len() - 1 {
                // ИСПРАВЛЕНИЕ: Мы имеем право искать слияние в ranks ТОЛЬКО если
                // обе соседние части состоят строго из 1 байта!
                if parts[i].len() == 1 && parts[i + 1].len() == 1 {
                    // Извлекаем чистые байты u8 из векторов
                    let b1 = parts[i][0];
                    let b2 = parts[i + 1][0];
                    let pair = (b1, b2); // Теперь тип строго (u8, u8)!

                    if let Some(&rank) = self.ranks.get(&pair) {
                        if rank < min_rank {
                            min_rank = rank;
                            best_pair = Some((i, pair));
                        }
                    }
                }
            }

            // Если нашли лучшую пару для слияния — схлопываем её
            if let Some((idx, _pair)) = best_pair {
                let mut first = parts.remove(idx);
                let second = parts.remove(idx);
                first.extend(second);
                parts.insert(idx, first);
            } else {
                break; // Если подходящих пар больше нет — выходим из цикла
            }
        }
        parts
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut final_token_ids = Vec::new();
        if text.is_empty() {
            final_token_ids.push(self.eos_token_id);
            return final_token_ids;
        }

        let raw_bytes = text.as_bytes().to_vec();
        let merged_parts = self.bpe_merge(raw_bytes);

        for part in merged_parts {
            if let Some(&id) = self.encoder.get(&part) {
                final_token_ids.push(id);
            }
        }

        final_token_ids.push(self.eos_token_id);
        final_token_ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut byte_buffer = Vec::new();

        for &id in ids {
            if id == self.eos_token_id {
                byte_buffer.extend_from_slice(b"<|endoftext|>");
                continue;
            }
            if let Some(bytes) = self.decoder.get(&id) {
                byte_buffer.extend_from_slice(bytes);
            }
        }

        String::from_utf8_lossy(&byte_buffer).into_owned()
    }
}
