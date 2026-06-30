use crate::utils::pinned_buffer::PinnedHostBuffer;
use crate::utils::{CudaBuffer, CudaStream};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

pub struct SafeTensorLoader {
    _file_mmap: Mmap,
    pub tensors: SafeTensors<'static>,
}

impl SafeTensorLoader {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap_ptr = mmap.as_ptr();
        let mmap_len = mmap.len();
        let static_slice = unsafe { std::slice::from_raw_parts(mmap_ptr, mmap_len) };

        let tensors = SafeTensors::deserialize(static_slice)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(SafeTensorLoader {
            _file_mmap: mmap,
            tensors,
        })
    }

    pub fn load_into_buffer(
        &self,
        tensor_name: &str,
        cuda_dst: &CudaBuffer,
        stream: &CudaStream,
    ) -> std::io::Result<()> {
        let view = self.tensors.tensor(tensor_name).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Критическая ошибка: Тензор '{}' не найден!", tensor_name),
            )
        })?;

        let raw_bytes = view.data();
        let cuda_len = cuda_dst.len();

        // 1. ПРОВЕРКА НА BF16 (Оригинальный формат Qwen3.5 на диске всегда 2 байта на элемент)
        let disk_bf16_elements = raw_bytes.len() / 2;

        // Берем минимум между тем, сколько есть на диске и сколько физически влезает в GPU буфер
        let target_len = cuda_len.min(disk_bf16_elements);

        let bf16_slice = unsafe {
            std::slice::from_raw_parts(raw_bytes.as_ptr() as *const u16, disk_bf16_elements)
        };

        // Создаем Pinned буфер строго под емкость целевого буфера GPU
        let mut pinned_buffer = PinnedHostBuffer::new(target_len);
        let fp32_slice = pinned_buffer.as_slice_mut();

        // Распаковываем на лету битовым сдвигом
        for i in 0..target_len {
            let bf16_val = bf16_slice[i];
            let fp32_bits = (bf16_val as u32) << 16;
            fp32_slice[i] = f32::from_bits(fp32_bits);
        }

        // Асинхронно отправляем по DMA на видеокарту
        cuda_dst.copy_from_host_async(fp32_slice, stream);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_safetensors_mmap_and_dma_upload_async() {
        let stream = CudaStream::new();
        let test_file_path = "target/debug/test_model.safetensors";

        // 1. Генерируем фейковый файл .safetensors для проверки математики на диске
        let weight_data = vec![0.1f32, 0.2, 0.3, 0.4];
        let byte_data = unsafe {
            std::slice::from_raw_parts(weight_data.as_ptr() as *const u8, weight_data.len() * 4)
        };

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.layers.0.attention.w_query.weight".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![2, 2], // Геометрия матрицы [2x2]
                byte_data,
            )
            .unwrap(),
        );

        // Сохраняем тестовые веса на диск в формате Hugging Face
        safetensors::tensor::serialize_to_file(
            &tensors,
            None::<std::collections::HashMap<String, String>>,
            Path::new(test_file_path),
        )
        .unwrap();

        // 2. Тестируем наш загрузчик: открываем через mmap
        let loader = SafeTensorLoader::open(test_file_path).unwrap();

        // Выделяем буфер на GPU
        let gpu_buffer = CudaBuffer::new(4);

        // Асинхронно загружаем веса по DMA
        loader
            .load_into_buffer(
                "model.layers.0.attention.w_query.weight",
                &gpu_buffer,
                &stream,
            )
            .unwrap();

        // 3. Скачиваем обратно для верификации
        let mut host_check = vec![0.0f32; 4];
        gpu_buffer.copy_to_host_async(&mut host_check, &stream);

        // Финальный барьер синхронизации
        stream.synchronize();

        // Сравниваем байты: математика должна совпасть до последнего знака
        assert_eq!(
            host_check,
            vec![0.1f32, 0.2, 0.3, 0.4],
            "Данные Safetensors повредились при DMA-трансфере!"
        );

        // Подчищаем тестовый файл с диска
        let _ = std::fs::remove_file(test_file_path);
        println!(
            "[ЮНИТ-ТЕСТ УСПЕШЕН] SafeTensorLoader идеально сопряжен с PinnedHostBuffer и CudaStream."
        );
    }
}
