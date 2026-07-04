use std::collections::HashMap;
use std::sync::atomic::Ordering;

pub struct CorpusAggregator;

impl CorpusAggregator {
    pub fn collect_unique_words(text: &str, cyrillic_regex: &str, num_threads: usize) -> (Vec<Vec<u32>>, Vec<u32>) {
        println!("[АГРЕГАТОР] Сборка уникальных цепочек через встроенный синтаксический движок regex-automata...");

        // Используем стандартный потокобезопасный Regex-поиск из regex_automata,
        // который у вас гарантированно уже есть в зависимостях
        let re = regex_automata::meta::Regex::new(cyrillic_regex)
            .expect("Невалидный паттерн регулярного выражения");

        let text_bytes = text.as_bytes();
        let chunk_size = (text_bytes.len() + num_threads - 1) / num_threads;

        let mut counts_map: HashMap<Vec<u32>, u32> = HashMap::with_capacity(65536);
        let counts_map_ptr = &mut counts_map as *mut HashMap<Vec<u32>, u32> as usize;

        let spin_lock = std::sync::atomic::AtomicBool::new(false);
        let spin_lock_ptr = &spin_lock as *const std::sync::atomic::AtomicBool as usize;

        std::thread::scope(|scope| {
            let counts_map_ref = counts_map_ptr;
            let lock_ref = spin_lock_ptr;

            for t_idx in 0..num_threads {
                let re_ref = &re;
                scope.spawn(move || {
                    // Выравниваем левую границу чанка по символам UTF-8
                    let mut start_pos = (t_idx * chunk_size).min(text_bytes.len());
                    while start_pos > 0 && start_pos < text_bytes.len() && !text.is_char_boundary(start_pos) {
                        start_pos += 1;
                    }

                    // Выравниваем правую границу чанка
                    let mut end_pos = ((t_idx + 1) * chunk_size).min(text_bytes.len());
                    while end_pos < text_bytes.len() && !text.is_char_boundary(end_pos) {
                        end_pos += 1;
                    }

                    if start_pos >= end_pos {
                        return;
                    }

                    let local_chunk = &text_bytes[start_pos..end_pos];
                    let mut local_counts: HashMap<Vec<u32>, u32> = HashMap::with_capacity(4096);

                    // Безопасно ищем совпадения регулярного выражения внутри чанка байт
                    for mat in re_ref.find_iter(local_chunk) {
                        let chunk_bytes = &local_chunk[mat.range()];
                        if chunk_bytes.is_empty() {
                            continue;
                        }
                        let word_u32: Vec<u32> = chunk_bytes.iter().map(|&b| b as u32).collect();
                        *local_counts.entry(word_u32).or_insert(0) += 1;
                    }

                    // Синхронизируем локальные результаты с глобальной картой через ваш spin-lock
                    unsafe {
                        let global_map = &mut *(counts_map_ref as *mut HashMap<Vec<u32>, u32>);
                        let atomic_lock = &*(lock_ref as *const std::sync::atomic::AtomicBool);

                        while atomic_lock
                            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                            .is_err()
                        {
                            std::hint::spin_loop();
                        }
                        for (k, v) in local_counts {
                            *global_map.entry(k).or_insert(0) += v;
                        }
                        atomic_lock.store(false, Ordering::Release);
                    }
                });
            }
        });

        println!("[ТРЕНЕР] Фаза агрегации завершена. Уникальных Qwen-цепочек: {}", counts_map.len());

        let (mut global_words, mut global_counts) = (Vec::with_capacity(counts_map.len()), Vec::with_capacity(counts_map.len()));
        for (w_u32, count) in counts_map {
            global_words.push(w_u32);
            global_counts.push(count);
        }

        (global_words, global_counts)
    }
}
