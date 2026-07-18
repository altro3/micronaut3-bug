[📥 src/kernels/input_stage/]
├── varlen_embeddings.cu           # Ядро #1: Сборка батча без PAD токенов в Ragged Tensors (Твоё `embeddings.cu` + `ragged_utils.cu`)
├── fused_rmsnorm_forward.cu       # Ядро #2: Стабилизация активаций слоев (Твоё `rmsnorm.cu` как fallback-версия)
└── fused_multimodal_projection.cu # Ядро #3: Прямая проекция и склейка патчей картинок/аудио на GPU

[⚡ src/kernels/prefill_stage/]
├── dequantize_fusion_mma.cu       # Ядро #4: Слитый GEMM (Q4_K) + распаковка в регистры на Tensor Cores (Твоё `dequantize_block_gguf_q4_k.cu`)
├── yarn_rope_scaling.cu           # Ядро #5: Поворот и динамическое YaRN-растяжение контекста (Твоё `rope.cu`)
├── flash_attention_prefill.cu     # Ядро #6: Compute-bound FA3 под Blackwell с поддержкой асинхронного TMA (Твоё `flash_attention_prefill_fp8.cu`)
├── chunked_prefill_splitter.cu    # Ядро #7: Нарезка промптов на чанки по 512 токенов
├── ring_attention_shifter.cu      # Ядро #8: P2P-пересылка блоков KV-кэша по кольцу для миллионного контекста
├── matmul_fp8_e4m3_fused.cu       # Ядро #46: Нативный FP8 GEMM под тензорные ядра Blackwell с Block-wise scaling
└── warpgroup_tma_swizzle.cu       # Ядро #47: Аппаратный swizzling памяти для 128-поточных варп-групп

[🔄 src/kernels/decoding_stage/]
├── flash_decoding_launcher.cu     # Ядро #9: Параллелизация редукции по длине контекста (Твой стек `flash_decoding_write/partial/combine/launcher.cu`)
├── paged_attention_indexer.cu     # Ядро #10: Виртуальные страницы + Sliding Window + Сдвиг адресов кэша
├── paged_kv_defrag.cu             # Ядро #11: Асинхронное уплотнение фрагментированной VRAM во время инференса
├── fused_mla_latent_reconstruct.cu# Ядро #12: Восстановление сжатого KV-кэша DeepSeek (MLA) в Shared Memory
└── matmul_fused_int2_int3_mma.cu  # Ядро #48: Суб-байтовая распаковка INT2/INT3 векторов через регистровые LUT

[🧬 src/kernels/mlp_moe_ssm/]
├── swiglu_fused.cu                # Ядро #13: SiLU + Умножение (MLP блок) в один проход без выгрузки в VRAM (Твоё `swiglu.cu` + `matmul.cu`)
├── moe_router_dispatcher.cu       # Ядро #14: Динамическое распределение токенов по экспертам (Scatter/Gather)
├── fused_gate_router.cu           # Ядро #15: Слитый MoE-гейт + Softmax без round-trip в глобальную память
└── fused_recurrent_scan.cu        # Ядро #16: Parallel Associative Scan для гибридных слоев Qwen 3.6 (Mamba/SSM)

[🦅 src/kernels/speculative_stage/]
├── multi_token_head.cu            # Ядро #17: Параллельный вывод независимых логических голов (Medusa/Eagle)
├── tree_attention_mask.cu         # Ядро #18: Древовидная маска причинности для валидации цепочек гипотез
├── fused_draft_verification.cu    # Ядро #19: Слитая проверка токенов черновой и большой моделей в один шаг (Твоё `speculative_verify.cu`)
├── speculative_tree_gather.cu     # Ядро #20: Схлопывание и валидация страниц KV после верификации дерева (Твоё `speculative_launcher.cu` + `speculative_sampling.cuh`)
└── speculative_tree_topk_filter.cu# Ядро #21: Многовариантный стохастический сэмплер по дереву условных вероятностей

[🎯 src/kernels/output_sampling/]
├── greedy_sampling.cu             # Ядро #22: Базовый детерминированный выбор максимального логита (Твоё `greedy_sampling.cu`)
└── gpu_stochastic_sampling.cu     # Ядро #23: Слитый топ-сэмплер (Top-K/Top-P/Min-P) строго на регистрах GPU (Твоё `fused_sampling.cu`)

[🚀 src/kernels/frontier_extensions/]
├── kv_cache_quantizer.cu          # Ядро #24: На лету сжатие K/V векторов в FP8/INT4 при записи в страницу (Твоё `prefill_kv_write.cu`)
├── cuda_graphs_monolith.cu        # Ядро #25: Упаковка статических цепочек ядер в монолитный граф CUDA
├── sink_token_window.cu           # Ядро #26: Удержание системного промпта (Attention Sink) при сдвиге окна
├── fused_rmsnorm_rope_kv_write.cu # Ядро #27: RMSNorm + RoPE + Запись в Paged KV в 1 проход (Ультимативный монолит)
├── pipeline_parallel_comm.cu      # Ядро #49: Асинхронный NCCL/P2P оверлап для работы в распределенном режиме
└── cross_attention_vit.cu         # Ядро #50: Кросс-внимание для полноценных Encoder-Decoder ViT моделей (Qwen VL)

[⚙️ src/kernels/system_runtime/]
├── inflight_batch_scheduler.cu    # Ядро #28: Врезка новых запросов на лету (Continuous Batching менеджера)
├── fused_bias_residual.cu         # Ядро #29: Слитое сложение Bias + Residual Add для перехода между слоями
├── kv_cache_host_swapper.cu       # Ядро #30: Асинхронный сброс страниц памяти в RAM хоста по PCIe 5.0 при OOM
├── beam_search_hypothesis.cu      # Ядро #31: Перестановка индексов страниц при многолучевом поиске
├── speculative_score_arbiter.cu   # Ядро #32: Вероятностный арбитраж принятия токенов-черновиков
└── tensor_zero_sanitizer.cu       # Ядро #33: Обнуление освобожденных страниц памяти (Безопасность/Изоляция)

[🧠 src/kernels/dynamic_reasoning/]
├── adaptive_layer_skipping.cu     # Ядро #34: Динамическая глубина (Early Exit) для пропуска слоев на простых токенах
├── cot_entropy_boundary_detector.cu# Ядро #35: Монитор энтропии мыслей для динамического управления Tree-Search в Qwen 3.6 Thinking
├── continuous_sliding_chunk_attention.cu # Ядро #36: Инкрементальное скользящее окно для real-time видео/аудио
├── speculative_pipelined_interleaver.cu  # Ядро #37: Асинхронный конвейер вычислений большой и драфт моделей
└── adaptive_context_router.cu     # Ядро #51: На лету переквантование старых страниц KV (FP16 -> INT4) при росте CoT

[☣️ src/kernels/black_ops_underground/]
├── zero_latency_persistent.cu     # Ядро #38: Автономный GPU-диспетчер (Persistent Kernel) без CPU-overhead
├── smart_allocation_speculator.cu # Ядро #39: Предикативное пре-выделение страниц под будущие токены
├── l2_locality_flash_decoding.cu  # Ядро #40: Распределение блоков с удержанием Attention Scores в L2-кэше
├── atomic_moe_balancer.cu         # Ядро #41: Атомарная балансировка токенов по экспертам без синхронизации с CPU
├── speculative_tree_pruning.cu    # Ядро #42: Раннее аппаратное отсечение ложных ветвей дерева в варпе
├── pcie5_prefetcher.cu            # Ядро #43: Асинхронный предвыборщик архивного контекста из ОЗУ процессора
├── zero_overhead_gc.cu            # Ядро #44: Теневой сборщик мусора VRAM внутри Warp Slack (свободные такты)
└── jit_warp_shuffler_optimizer.cu # Ядро #45: На лету автотюнинг размеров плиток и шаффла под батч и Blackwell
