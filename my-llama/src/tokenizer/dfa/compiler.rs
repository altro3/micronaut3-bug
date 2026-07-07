use regex_automata::dfa::dense::{Builder, Config, DFA};
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID;
use std::fs::File;
use std::io::{BufWriter, Result, Write};

pub struct DfaCompiler;

impl DfaCompiler {
    // Сигнатура метода строго сохранена
    pub fn compile_qwen_dfa(raw_qwen_regex: &str, trans_path: &str, accept_path: &str) -> Result<()> {
        println!("[DFA-КОМПИЛЯТОР] Построение детерминированного автомата (DFA) из динамического паттерна...");

        let is_stock_qwen = raw_qwen_regex.contains("(?i:'s|'t|'re|'ve|'m|'ll|'d)");

        let qwen_regex = if is_stock_qwen {
            println!("[DFA-КОМПИЛЯТОР] Обнаружен стандартный англоязычный паттерн. Адаптируем под кириллический максимум...");
            r" ?\p{L}+|\p{L}+| ?\p{N}+|[^\s\p{L}\p{N}]+|\s*[\r\n]+|\s+".to_string()
        } else {
            raw_qwen_regex.replace(r"\s+(?!\S)", r"\s+").replace(r"\s+(?!\\S)", r"\s+")
        };

        let syntax_config = regex_automata::util::syntax::Config::new().unicode(true).utf8(true);

        let dfa_config = Config::new().minimize(true).byte_classes(true);

        let dfa: DFA<Vec<u32>> = Builder::new()
            .syntax(syntax_config)
            .configure(dfa_config)
            .build(&qwen_regex)
            .unwrap_or_else(|e| {
                panic!("Критическая ошибка компиляции паттерна: {:?}\nПаттерн: {}", e, qwen_regex);
            });

        println!("[DFA-КОМПИЛЯТОР] Сериализация матрицы переходов...");
        let mut dfa_bytes = vec![0u8; dfa.write_to_len()];
        dfa.write_to_little_endian(&mut dfa_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("DFA Serializer Error: {:?}", e)))?;

        let stride = dfa.stride();
        let total_states_allocated = dfa_bytes.len() / stride;
        let mut accept_bytes = vec![0u8; total_states_allocated];

        println!("[DFA-КОМПИЛЯТОР] Сборка карты принимающих состояний...");
        for (state_idx, slot) in accept_bytes.iter_mut().enumerate() {
            let raw_offset = state_idx * stride;
            let Ok(state_id) = StateID::new(raw_offset) else { continue };
            if dfa.is_match_state(state_id) {
                *slot = 1;
            }
        }

        let mut trans_file = BufWriter::new(File::create(trans_path)?);
        trans_file.write_all(&dfa_bytes)?;

        let mut accept_file = BufWriter::new(File::create(accept_path)?);
        accept_file.write_all(&accept_bytes)?;

        println!(
            "[DFA-КОМПИЛЯТОР] УСПЕХ! Скомпилировано состояний: {}. Размер файла: {} байт.",
            total_states_allocated,
            dfa_bytes.len()
        );
        Ok(())
    }
}
