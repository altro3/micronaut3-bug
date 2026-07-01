use crate::cuda::{CudaBuffer, CudaStream};
use crate::models::compiler::llama::LlamaGraphCompiler;
use crate::models::{ModelGraph, UniversalComputationGraph};
use crate::utils::parameter::{DataType, Parameter};
use serde_json::{from_reader, Value};
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Result};

pub struct ModelFactory;

impl ModelFactory {
    pub fn create_from_config(
        config_path: &str,
        weights_path: &str,
        batch_size: usize,
        is_training: bool,
        stream: &CudaStream,
    ) -> Result<Box<dyn ModelGraph>> {
        let file = File::open(config_path)?;
        let reader = BufReader::new(file);
        let config: Value = from_reader(reader).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        let model_type = config["model_type"].as_str().unwrap_or("llama");
        let hidden_size = config["hidden_size"]
            .as_u64()
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "hidden_size не найден в конфиге"))? as usize;
        let num_layers = config["num_hidden_layers"]
            .as_u64()
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "num_hidden_layers не найден"))? as usize;
        let intermediate_size = config["intermediate_size"]
            .as_u64()
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "intermediate_size не найден"))? as usize;
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;

        let num_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as i32;
        let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap_or(num_heads as u64) as i32;
        let head_dim = (hidden_size / num_heads as usize) as i32;

        let dtype = match config["torch_dtype"].as_str() {
            Some("bfloat16") => DataType::BF16,
            Some("float16") => DataType::F16,
            _ => DataType::F32,
        };

        let compiled = match model_type {
            "llama" | "qwen2" => LlamaGraphCompiler::compile(
                num_layers,
                hidden_size,
                intermediate_size,
                vocab_size,
                rms_norm_eps,
                dtype,
                batch_size,
                num_heads,
                num_kv_heads,
                head_dim,
                is_training,
            ),
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    format!("Архитектурный граф для '{}' еще не реализован в компиляторе", model_type),
                ));
            }
        };

        // Аллокация тензоров весов на GPU
        let mut weights = Vec::with_capacity(compiled.weight_specs.len());
        for spec in &compiled.weight_specs {
            let param = Parameter::new(spec.shape.clone(), dtype, is_training, stream);
            weights.push(param);
        }

        // Загрузка сырых параметров из SafeTensors на хосте
        crate::models::llama_loader::load_model_weights(
            weights_path,
            &compiled.weight_specs,
            &mut weights,
            stream,
        )
            .map_err(|e| Error::new(ErrorKind::Other, format!("Критическая ошибка загрузки весов: {}", e)))?;

        let allocation_units = (compiled.max_arena_bytes + 3) / 4;
        let activation_arena = CudaBuffer::new(allocation_units);

        println!(
            "[PROD-FACTORY] Монолитный вычислительный граф успешно инициализирован:\n\
             |-> Архитектура: '{}' ({:?})\n\
             |-> Режим работы: {}\n\
             |-> Выделено памяти под веса (кол-во тензоров): {}\n\
             |-> Размер непрерывной арены активаций в VRAM: {} байт\n\
             |-> Количество скомпилированных GPU инструкций: {}",
            model_type,
            dtype,
            if is_training { "ОБУЧЕНИЕ (Градиенты активны)" } else { "ИНФЕРЕНС (Энергосберегающий)" },
            weights.len(),
            compiled.max_arena_bytes,
            compiled.pipeline.len()
        );

        Ok(Box::new(UniversalComputationGraph {
            weights,
            activation_arena,
            pipeline: compiled.pipeline,
            logits_tensor: compiled.logits_tensor,
            targets_offset: compiled.targets_offset,
        }))
    }
}
