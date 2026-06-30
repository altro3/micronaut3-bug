use crate::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {

    fn launch_update_kv_cache(
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        new_k: *const c_void,
        new_v: *const c_void,
        token_index: i32,
        hidden_size: i32,
        stream: *mut c_void,
    );
}

pub struct KvCache<'a> {
    pub k_cache: CudaBuffer,
    pub v_cache: CudaBuffer,
    pub current_len: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> KvCache<'a> {
    pub fn len(&self) -> usize {
        self.current_len
    }
}

pub struct KvCacheManager {
    pub k_storage: CudaBuffer,
    pub v_storage: CudaBuffer,
    pub max_seq_len: usize,
    pub hidden_size: usize,
    current_pos: usize,
}

impl KvCacheManager {
    pub fn new(max_seq_len: usize, hidden_size: usize) -> Self {
        let total_elements = max_seq_len * hidden_size;
        let k_storage = CudaBuffer::new(total_elements);
        let v_storage = CudaBuffer::new(total_elements);

        KvCacheManager {
            k_storage,
            v_storage,
            max_seq_len,
            hidden_size,
            current_pos: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.current_pos
    }

    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }

    pub fn append_async(
        &mut self,
        new_key: &CudaBuffer,
        new_value: &CudaBuffer,
        stream: &CudaStream,
    ) {
        if self.is_full() {
            panic!(
                "Критическая ошибка: Превышен максимальный лимит контекста KV-Cache ({})!",
                self.max_seq_len
            );
        }

        let t_index = self.current_pos as i32;
        let h_size = self.hidden_size as i32;

        unsafe {
            launch_update_kv_cache(
                self.k_storage.as_raw_ptr(),
                self.v_storage.as_raw_ptr(),
                new_key.as_raw_ptr(),
                new_value.as_raw_ptr(),
                t_index,
                h_size,
                stream.as_raw(),
            );
        }

        self.current_pos += 1;
    }

    pub fn get_view(&self) -> KvCache<'_> {
        let active_elements = self.current_pos * self.hidden_size;

        KvCache {
            k_cache: self.k_storage.slice(0, active_elements),
            v_cache: self.v_storage.slice(0, active_elements),
            current_len: self.current_pos,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn clear_async(&mut self, stream: &CudaStream) {
        self.k_storage.zero_out_async(stream);
        self.v_storage.zero_out_async(stream);
        self.current_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_manager_lifecycle_async() {
        const HIDDEN_SIZE: usize = 3;
        const MAX_SEQ_LEN: usize = 5;

        let stream = CudaStream::new();

        // 1. Инициализируем менеджер кэша
        let mut kv_manager = KvCacheManager::new(MAX_SEQ_LEN, HIDDEN_SIZE);
        assert_eq!(kv_manager.len(), 0);
        assert!(!kv_manager.is_full());

        // 2. Создаем временные буферы под один новый прилетевший токен
        let token_k = CudaBuffer::new(HIDDEN_SIZE);
        let token_v = CudaBuffer::new(HIDDEN_SIZE);

        // ИСПРАВЛЕНИЕ 1: Явно размечаем тип f32 для векторов
        token_k.copy_from_host_async(&vec![10.0f32, 20.0, 30.0], &stream);
        token_v.copy_from_host_async(&vec![0.1f32, 0.2, 0.3], &stream);

        // 3. Добавляем токен в кэш через стрим
        kv_manager.append_async(&token_k, &token_v, &stream);
        assert_eq!(kv_manager.len(), 1);

        // 4. Проверяем генерацию View-слайса
        let view = kv_manager.get_view();
        assert_eq!(view.len(), 1);
        assert_eq!(view.k_cache.len(), HIDDEN_SIZE);

        // ИСПРАВЛЕНИЕ 2: Исправляем имя переменной на kv_manager
        assert_eq!(view.k_cache.as_raw_ptr(), kv_manager.k_storage.as_raw_ptr());

        // 5. Тестируем асинхронную очистку
        kv_manager.clear_async(&stream);
        assert_eq!(kv_manager.len(), 0);

        // Финальная синхронизация
        stream.synchronize();
        println!("[ЮНИТ-ТЕСТ УСПЕШЕН] Декомпозиция KV-Cache на Manager и View работает идеально.");
    }
}
