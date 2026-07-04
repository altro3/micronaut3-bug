use super::decoder::HfByteDecoder;
use super::types::{CompiledVocabulary, FlatTrieNode, QwenJsonModel};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Result};

struct BuilderNode {
    token_id: u32,
    children: [u32; 256],
}

pub struct DictCompiler;

impl DictCompiler {
    pub fn compile_from_json(file_path: &str) -> Result<CompiledVocabulary> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let root: QwenJsonModel = serde_json::from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        let eos_token_id = Self::extract_eos(&root);
        let vocab_size = root.model.vocab.len();

        let mut byte_fallback = [u32::MAX; 256];
        let mut vocab_compiled_tokens = vec![Vec::new(); vocab_size.max(260000)];
        for (token_str, &id) in root.model.vocab.iter() {
            let raw_bytes = HfByteDecoder::decode_string(token_str);
            let id_idx = id as usize;

            if id_idx >= vocab_compiled_tokens.len() {
                vocab_compiled_tokens.resize(id_idx + 1, Vec::new());
            }
            vocab_compiled_tokens[id_idx] = raw_bytes.clone();

            if raw_bytes.len() == 1 {
                byte_fallback[raw_bytes[0] as usize] = id;
            }
        }

        Self::fill_qwen_byte_fallbacks(&root, &mut byte_fallback);

        println!("[ДИАГНОСТИКА TRIE] Начинаю BFS-сборку дерева. Всего токенов в словаре: {}", vocab_size);

        let mut trie_root_offsets = [u32::MAX; 256];
        let mut trie_nodes = Vec::with_capacity(vocab_size * 2);

        // Корень сырого дерева-черновика всегда под индексом 0
        let mut builder_nodes = vec![BuilderNode {
            token_id: u32::MAX,
            children: [u32::MAX; 256],
        }];

        let mut inserted_tokens = 0;
        for id in 0..vocab_size {
            let bytes = &vocab_compiled_tokens[id];
            if bytes.is_empty() {
                continue;
            }

            let mut curr_node_idx = 0usize;

            for &byte in bytes.iter() {
                let b = byte as usize;
                let next_idx = builder_nodes[curr_node_idx].children[b];

                if next_idx == u32::MAX {
                    let new_idx = builder_nodes.len() as u32;
                    builder_nodes.push(BuilderNode {
                        token_id: u32::MAX,
                        children: [u32::MAX; 256],
                    });
                    builder_nodes[curr_node_idx].children[b] = new_idx;
                    curr_node_idx = new_idx as usize;
                } else {
                    curr_node_idx = next_idx as usize;
                }
            }
            builder_nodes[curr_node_idx].token_id = id as u32;
            inserted_tokens += 1;
        }
        println!("[ДИАГНОСТИКА TRIE] В дерево-черновик добавлено токенов: {}", inserted_tokens);

        // --- ЛИНЕАРИЗАЦИЯ ЧЕРЕЗ BFS ОЧЕРЕДЬ (Гарантия 0 дубликатов) ---
        // Очередь хранит пары: (индекс_в_builder_nodes, индекс_в_trie_nodes)
        let mut queue = VecDeque::new();
        let mut active_roots = 0;

        // Инициализируем первый уровень дерева (корневые переходы по первому байту)
        for b in 0..256 {
            let child_builder_idx = builder_nodes[0].children[b];
            if child_builder_idx != u32::MAX {
                let flat_idx = trie_nodes.len() as u32;
                trie_root_offsets[b] = flat_idx;
                active_roots += 1;

                trie_nodes.push(FlatTrieNode {
                    token_id: builder_nodes[child_builder_idx as usize].token_id,
                    children_offset: u32::MAX,
                });

                queue.push_back((child_builder_idx as usize, flat_idx as usize));
            }
        }

        // Обходим граф по слоям
        while let Some((b_idx, f_idx)) = queue.pop_front() {
            let b_node = &builder_nodes[b_idx];

            let mut has_children = false;
            for &c in b_node.children.iter() {
                if c != u32::MAX {
                    has_children = true;
                    break;
                }
            }

            if has_children {
                // Выделяем сплошной блок из 256 слотов под детей текущего узла
                let children_offset = trie_nodes.len() as u32;
                trie_nodes.resize(
                    trie_nodes.len() + 256,
                    FlatTrieNode {
                        token_id: u32::MAX,
                        children_offset: u32::MAX,
                    },
                );

                // Привязываем смещение детей к родителю
                trie_nodes[f_idx].children_offset = children_offset;

                for b in 0..256 {
                    let child_builder_idx = b_node.children[b];
                    if child_builder_idx != u32::MAX {
                        let target_flat_slot = (children_offset as usize) + b;

                        // Записываем точные данные ребенка
                        trie_nodes[target_flat_slot].token_id = builder_nodes[child_builder_idx as usize].token_id;

                        // Пушим ребенка в очередь для обработки его поддеревьев на следующем слое
                        queue.push_back((child_builder_idx as usize, target_flat_slot));
                    }
                }
            }
        }

        // Финальная валидация плоского дерева
        let mut flattened_valid_tokens = 0;
        for node in trie_nodes.iter() {
            if node.token_id != u32::MAX {
                flattened_valid_tokens += 1;
            }
        }

        println!("[ДИАГНОСТИКА TRIE] BFS-линеаризация завершена.");
        println!("  ├── Активных корневых переходов (1-й байт): {}", active_roots);
        println!("  ├── Финальный размер массива trie_nodes: {}", trie_nodes.len());
        println!("  └── Валидных ID токенов в плоском Trie: {}", flattened_valid_tokens);
        println!("======================================================================");

        let raw_pairs = Vec::new();
        let extracted_regex = root
            .pre_tokenizer
            .pretokenizers
            .first()
            .and_then(|entry| entry.pattern.as_ref())
            .map(|p| p.regex.clone())
            .unwrap_or_else(|| r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]+|\p{L}+|\p{N}{1,3}".to_string());

        Ok(CompiledVocabulary {
            byte_fallback,
            raw_pairs,
            eos_token_id,
            vocab_size,
            vocab_compiled_tokens,
            extracted_regex,
            trie_nodes,
            trie_root_offsets,
        })
    }

    fn extract_eos(root: &QwenJsonModel) -> u32 {
        if let Some(ref added) = root.added_tokens {
            for tok in added {
                if tok.content == "<|endoftext|>" {
                    return tok.id;
                }
            }
        }
        248044
    }

    fn fill_qwen_byte_fallbacks(root: &QwenJsonModel, fallback: &mut [u32; 256]) {
        for b in 0..=255 {
            if fallback[b] == u32::MAX {
                let byte_token_name = format!("<|byte_{:02X}|>", b);
                if let Some(&id) = root.model.vocab.get(&byte_token_name) {
                    fallback[b] = id;
                } else {
                    fallback[b] = (root.model.vocab.len() as u32) + (b as u32);
                }
            }
        }
    }
}
