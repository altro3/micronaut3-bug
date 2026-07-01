use crate::cuda::CudaStream;
use crate::models::compiler::llama::WeightSpec;
use crate::utils::parameter::Parameter;
use memmap2::Mmap;
use std::fs::File;
use std::io::{Error, ErrorKind, Result};

pub fn load_model_weights(weights_path: &str, weight_specs: &[WeightSpec], weights: &mut [Parameter], stream: &CudaStream) -> Result<()> {
    let file = File::open(weights_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    println!("[LOADER] Файл весов успешно спроецирован через mmap. Размер: {} байт", mmap.len());

    let json_len_bytes = mmap
        .get(0..8)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Файл слишком мал или поврежден"))?;

    let json_len = u64::from_le_bytes(json_len_bytes.try_into().unwrap()) as usize;

    let json_slice = mmap
        .get(8..(8 + json_len))
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Неверный размер заголовка Safetensors"))?;

    let metadata: serde_json::Value =
        serde_json::from_slice(json_slice).map_err(|e| Error::new(ErrorKind::InvalidData, format!("Ошибка парсинга JSON Safetensors: {}", e)))?;

    let data_start_offset = 8 + json_len;

    for (idx, spec) in weight_specs.iter().enumerate() {
        let tensor_info = &metadata[&spec.name];

        if tensor_info.is_null() {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!("Критическая ошибка: Тензор '{}' не найден в файле модели!", spec.name),
            ));
        }

        let data_offsets = tensor_info["data_offsets"]
            .as_array()
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, format!("Отсутствуют data_offsets для {}", spec.name)))?;

        let start_byte = data_offsets[0].as_u64().unwrap() as usize;
        let end_byte = data_offsets[1].as_u64().unwrap() as usize;
        let tensor_bytes_count = end_byte - start_byte;

        let file_tensor_start = data_start_offset + start_byte;
        let file_tensor_slice = mmap
            .get(file_tensor_start..(file_tensor_start + tensor_bytes_count))
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, format!("Выход за границы файла при чтении {}", spec.name)))?;

        let gpu_parameter = &weights[idx];

        gpu_parameter.data.copy_from_host_slice(file_tensor_slice, stream);
    }

    stream.synchronize();
    println!(
        "[LOADER] Все {} тензоров весов успешно загружены в VRAM по технологии Zero-Copy.",
        weight_specs.len()
    );

    Ok(())
}
