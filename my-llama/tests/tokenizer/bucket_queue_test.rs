#[cfg(test)]
mod tests {
    use my_llama::tokenizer::bucket_queue::BucketQueue;

    #[test]
    fn test_bucket_queue_hardcore_bpe_flow() {
        let vocab_size = 10000;
        let max_chunk_capacity = 500;
        // queue теперь имеет тип Box<BucketQueue>
        let mut queue = BucketQueue::with_capacity(vocab_size, max_chunk_capacity);

        queue.push(100, 10);
        queue.push(50, 5);
        queue.push(500, 50);
        queue.push(50, 6);

        assert_eq!(queue.min_rank_dirty, 50);
        assert_eq!(queue.max_rank_dirty, 500);

        let packed1 = queue.pop_packed();
        assert_ne!(packed1, u64::MAX);
        assert_eq!((packed1 >> 32) as u32, 50);
        assert_eq!((packed1 & 0xFFFFFFFF) as usize, 6);

        let packed2 = queue.pop_packed();
        assert_eq!((packed2 >> 32) as u32, 50);
        assert_eq!((packed2 & 0xFFFFFFFF) as usize, 5);

        let packed3 = queue.pop_packed();
        assert_eq!((packed3 >> 32) as u32, 100);
        assert_eq!((packed3 & 0xFFFFFFFF) as usize, 10);

        let packed4 = queue.pop_packed();
        assert_eq!((packed4 >> 32) as u32, 500);
        assert_eq!((packed4 & 0xFFFFFFFF) as usize, 50);

        assert_eq!(queue.pop_packed(), u64::MAX);

        // Тест очистки
        queue.push(200, 20);
        queue.push(8000, 80);

        // Передаем 100, так как максимальный индекс ноды был 80
        queue.clear(100);

        assert_eq!(queue.min_rank_dirty, queue.buckets.len());
        assert_eq!(queue.pop_packed(), u64::MAX);
    }
}
