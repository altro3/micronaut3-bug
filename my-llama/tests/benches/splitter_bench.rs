use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use my_llama::tokenizer::SimdSplitter;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

fn bench_native_multithreaded_splitter(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdSplitter_Native_Threads");

    // Суровый мультиязычный промпт
    let multilang_sample = "Привет! Ты — мощная языковая модель Llama-3.1-v2.\n\
    \tdef train_fast_tokenizer(data: &[u8]) -> usize {\n\
    \t    let speed = \"космическая скорость\"; // SIMD-алгоритмы помогают!\n\
    \t    println!(\"Performance: {}, CPU: Arrow Lake\", speed);\n\
    \t}\n\
    Unbelievable industrial-grade artificial intelligence performance test on Rust.\n\
    === Юникод тест: 🚀🤖🔥 ===\n\
    === Азиатский тест (Иероглифы): 人工智能 / 大语言模型 / 速度 ===\n\
    \n\
    \t\t[SYSTEM OVERRIDE]: active_mode = true;\r\n  ";

    // Увеличиваем объем данных до ~170 МБ для полной загрузки контроллера памяти
    let total_cores = 16; // 16 чистых физических ядер Ultra 9 285H
    let iterations_per_core = 20_000;

    // Генерируем 16 независимых тяжелых кусков текста (по одному для каждого потока)
    let mut chunks_dataset: Vec<String> = Vec::with_capacity(total_cores);
    for _ in 0..total_cores {
        let mut chunk_string = String::with_capacity(multilang_sample.len() * iterations_per_core);
        for _ in 0..iterations_per_core {
            chunk_string.push_str(multilang_sample);
        }
        chunks_dataset.push(chunk_string);
    }

    let total_bytes_len: usize = chunks_dataset.iter().map(|s| s.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes_len as u64));
    let dataset_size_mb = format!("{:.2} MB", total_bytes_len as f64 / 1_048_576.0);

    // === 1. ПОСЛЕДОВАТЕЛЬНЫЙ ЗАМЕР (1 ЯДРО) ===
    group.bench_function(BenchmarkId::new("single_thread", &dataset_size_mb), |b| {
        // Выделяем плоские буферы под размер одного чанка
        let chunk_len = chunks_dataset[0].len();
        let mut offsets_buffer = vec![0u32; chunk_len + 64];
        let mut lens_buffer = vec![0u32; chunk_len + 64];

        b.iter(|| {
            for text_chunk in black_box(&chunks_dataset) {
                let count = SimdSplitter::split(black_box(text_chunk), &mut offsets_buffer, &mut lens_buffer);
                black_box(count);
            }
        });
    });

    // === 2. МНОГОПОТОЧНЫЙ ЗАМЕР НА 16 РОДНЫХ ПОТОКАХ (БЕЗ RAYON) ===
    // Барьеры для ультрабыстрой синхронизации фаз потоков без оверхеда ОС
    let start_barrier = Arc::new(Barrier::new(total_cores + 1));
    let end_barrier = Arc::new(Barrier::new(total_cores + 1));
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    // Создаем пул долгоживущих потоков перед стартом итераций
    let mut thread_handles = Vec::with_capacity(total_cores);

    for core_id in 0..total_cores {
        let text_chunk = chunks_dataset[core_id].clone();
        let t_start_barrier = Arc::clone(&start_barrier);
        let t_end_barrier = Arc::clone(&end_barrier);
        let t_shutdown = Arc::clone(&shutdown_flag);

        let handle = thread::spawn(move || {
            // Выделяем персональные буферы потока один раз при старте!
            // Никаких аллокаций в цикле — это гарантирует максимальный FPS.
            let mut local_offsets = vec![0u32; text_chunk.len() + 64];
            let mut local_lens = vec![0u32; text_chunk.len() + 64];

            loop {
                // Фаза 1: Ждем команду на старт от главного потока Criterion
                t_start_barrier.wait();

                // Проверяем, не пора ли гасить пул потоков
                if t_shutdown.load(Ordering::Relaxed) {
                    break;
                }

                // Вычисление: Ядро молотит свой персональный кусок текста через AVX2
                let count = SimdSplitter::split(black_box(&text_chunk), &mut local_offsets, &mut local_lens);
                black_box(count);

                // Фаза 2: Сигнализируем главному потоку, что ядро закончило вычисления
                t_end_barrier.wait();
            }
        });
        thread_handles.push(handle);
    }

    group.bench_function(BenchmarkId::new("native_threads_16_cores", &dataset_size_mb), |b| {
        b.iter(|| {
            // 1. Даем одновременный старт всем 16 потокам на ядрах Ultra 9
            start_barrier.wait();

            // В этот момент все 16 ядер параллельно щелкают SIMD-инструкции...

            // 2. Ждем, пока последнее 16-е ядро отрапортует о финише
            end_barrier.wait();
        });
    });

    // Корректно тушим пул потоков после окончания всех бенчмарков
    shutdown_flag.store(true, Ordering::Relaxed);
    start_barrier.wait(); // Освобождаем потоки из цикла loop, чтобы они увидели флаг shutdown

    for handle in thread_handles {
        let _ = handle.join();
    }

    group.finish();
}

criterion_group!(benches, bench_native_multithreaded_splitter);
criterion_main!(benches);
