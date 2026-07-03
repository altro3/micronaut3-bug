#[cfg(test)]
mod tests {
    use my_llama::tokenizer::bucket_queue::BucketQueue;

    #[test]
    fn test_bucket_queue_hardcore_bpe_flow() {
        let vocab_size = 10000;
        let max_chunk_capacity = 500;
        let mut queue = BucketQueue::with_capacity(vocab_size, max_chunk_capacity);

        // --- Сценарий 1: Базовый упорядоченный pop ---
        queue.push(100, 10);
        queue.push(50, 5);
        queue.push(500, 50);
        queue.push(50, 6);

        assert_eq!(queue.min_rank_dirty, 50);
        assert_eq!(queue.max_rank_dirty, 500);

        let packed1 = queue.pop_packed();
        assert_ne!(packed1, u64::MAX);
        let rank1 = (packed1 >> 32) as u32;
        let idx1 = (packed1 & 0xFFFFFFFF) as usize;
        assert_eq!(rank1, 50);
        assert_eq!(idx1, 6);

        let packed2 = queue.pop_packed();
        let rank2 = (packed2 >> 32) as u32;
        let idx2 = (packed2 & 0xFFFFFFFF) as usize;
        assert_eq!(rank2, 50);
        assert_eq!(idx2, 5);

        let packed3 = queue.pop_packed();
        let rank3 = (packed3 >> 32) as u32;
        let idx3 = (packed3 & 0xFFFFFFFF) as usize;
        assert_eq!(rank3, 100);
        assert_eq!(idx3, 10);

        let packed4 = queue.pop_packed();
        let rank4 = (packed4 >> 32) as u32;
        let idx4 = (packed4 & 0xFFFFFFFF) as usize;
        assert_eq!(rank4, 500);
        assert_eq!(idx4, 50);

        assert_eq!(queue.pop_packed(), u64::MAX);

        // --- Сценарий 2: Инкрементальный Clear по грязным регионам ---
        queue.push(200, 20);
        queue.push(8000, 80);

        queue.clear(100);

        assert_eq!(queue.min_rank_dirty, queue.buckets.len());
        assert_eq!(queue.pop_packed(), u64::MAX);

        let start_word = 200 >> 6;
        let end_word = 8000 >> 6;
        for word_idx in start_word..=end_word {
            assert_eq!(queue.bitset[word_idx], 0, "Bitset word {} was not cleared!", word_idx);
        }

        // --- Сценарий 3: Стресс-тест AVX2/BMI2 поиска ---
        queue.push(10, 1);
        queue.push(65, 2);
        queue.push(300, 3);

        let p_10 = queue.pop_packed();
        assert_eq!((p_10 >> 32) as u32, 10);
        assert_eq!((p_10 & 0xFFFFFFFF) as usize, 1);

        let p_65 = queue.pop_packed();
        assert_eq!((p_65 >> 32) as u32, 65);
        assert_eq!((p_65 & 0xFFFFFFFF) as usize, 2);

        let p_300 = queue.pop_packed();
        assert_eq!((p_300 >> 32) as u32, 300);
        assert_eq!((p_300 & 0xFFFFFFFF) as usize, 3);

        assert_eq!(queue.pop_packed(), u64::MAX);
    }
}
