/*use crate::cuda::CudaStream;
use crate::models::compiler::llama::WeightSpec;
use crate::utils::parameter::{DataType, Parameter};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use std::slice::from_raw_parts;

pub struct SafeTensorLoader {
    _file_mmap: Mmap,
    pub tensors: SafeTensors<'static>,
}

impl SafeTensorLoader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap_ptr = mmap.as_ptr();
        let mmap_len = mmap.len();
        let static_slice = unsafe { from_raw_parts(mmap_ptr, mmap_len) };

        let tensors =
            SafeTensors::deserialize(static_slice).map_err(|e| Error::new(ErrorKind::InvalidData, format!("Ошибка парсинга Safetensors: {}", e)))?;

        Ok(SafeTensorLoader { _file_mmap: mmap, tensors })
    }
}

pub fn load_model_weights(weights_path: &str, weight_specs: &[WeightSpec], weights: &mut [Parameter], stream: &CudaStream) -> Result<()> {
    let loader = SafeTensorLoader::open(weights_path)?;

    for (idx, spec) in weight_specs.iter().enumerate() {
        let view = loader
            .tensors
            .tensor(&spec.name)
            .map_err(|_| Error::new(ErrorKind::NotFound, format!("Тензор '{}' не найден в файле весов!", spec.name)))?;

        let raw_bytes = view.data();
        let gpu_parameter = &weights[idx];

        if gpu_parameter.dtype != DataType::F32 {
            gpu_parameter.data.copy_from_host_slice(raw_bytes, stream);
            continue;
        }

        let disk_bf16_elements = raw_bytes.len() / 2;
        let bf16_slice = unsafe { from_raw_parts(raw_bytes.as_ptr() as *const u16, disk_bf16_elements) };

        let mut fp32_vector = vec![0.0f32; disk_bf16_elements];
        fp32_vector.par_iter_mut().enumerate().for_each(|(i, fp32_val)| {
            let bf16_val = bf16_slice[i];
            let fp32_bits = (bf16_val as u32) << 16;
            *fp32_val = f32::from_bits(fp32_bits);
        });

        gpu_parameter.data.copy_from_host_slice(&fp32_vector, stream);
    }

    stream.synchronize();
    println!("[LOADER] Все веса успешно импортированы в VRAM конвейером нулевого копирования.");

    Ok(())
}
*/