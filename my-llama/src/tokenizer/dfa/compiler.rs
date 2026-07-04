use regex_automata::dfa::dense::{Builder, Config, DFA};
use regex_automata::dfa::Automaton;
use std::fs::File;
use std::io::{BufWriter, Result, Write};

pub struct DfaCompiler;

impl DfaCompiler {
    pub fn compile_qwen_dfa(raw_qwen_regex: &str, trans_path: &str, accept_path: &str) -> Result<()> {
        println!("[DFA-КОМПИЛЯТОР] Построение детерминированного автомата (DFA) из динамического паттерна...");

        let qwen_regex = raw_qwen_regex.replace(r"\s+(?!\S)", r"\s+").replace(r"\s+(?!\S)", r"\s+");

        let dfa_config = Config::new().minimize(true).byte_classes(false);

        let dfa: DFA<Vec<u32>> = Builder::new().configure(dfa_config).build(&qwen_regex).unwrap_or_else(|e| {
            panic!("Критическая ошибка компиляции очищенного паттерна: {:?}\nПаттерн: {}", e, qwen_regex);
        });

        println!("[DFA-КОМПИЛЯТОР] Сериализация матрицы переходов...");
        let mut dfa_bytes = vec![0u8; dfa.write_to_len()];
        dfa.write_to_little_endian(&mut dfa_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("DFA Serializer Error: {:?}", e)))?;

        let stride = dfa.stride();
        let total_states_allocated = dfa_bytes.len() / stride;
        let mut accept_bytes = vec![0u8; total_states_allocated];

        println!("[DFA-КОМПИЛЯТОР] Сборка карты принимающих состояний...");
        for state_idx in 0..total_states_allocated {
            let raw_offset = state_idx * stride;
            if let Ok(state_id) = regex_automata::util::primitives::StateID::new(raw_offset) {
                if dfa.is_match_state(state_id) {
                    accept_bytes[state_idx] = 1;
                }
            }
        }

        let mut trans_file = BufWriter::new(File::create(trans_path)?);
        trans_file.write_all(&dfa_bytes)?;

        let mut accept_file = BufWriter::new(File::create(accept_path)?);
        accept_file.write_all(&accept_bytes)?;

        println!("[DFA-КОМПИЛЯТОР] УСПЕХ! Бинарный кэш уплощенного DFA успешно сохранен.");
        Ok(())
    }
}
