use crate::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_update_kv_cache(
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        new_k: *const c_void,
        new_v: *const c_void,
        token_index: i32,
        hidden_size: i32,
    );
}

/// Промышленный буфер кэширования Ключей и Значений (KV-Cache) для My-Qwen
pub struct KvCache {
    // Непрерывный буфер для векторов Ключей (Key) во VRAM
    pub k_cache: CudaBuffer,
    // Непрерывный буфер для векторов Значений (Value) во VRAM
    pub v_cache: CudaBuffer,

    // Максимально допустимый размер контекста (например, 2048 или 4096 токенов)
    max_seq_len: usize,
    // Размерность вектора одного токена (hidden_size)
    hidden_size: usize,
    // Текущая позиция в кэше (сколько токенов уже обработано и записано)
    current_pos: usize,
}

impl KvCache {
    /// Конструктор: сразу выделяет фиксированный объем VRAM под максимальный контекст.
    /// В LLM память под KV-Cache выделяется один раз при старте, чтобы избежать фрагментации памяти GPU.
    pub fn new(max_seq_len: usize, hidden_size: usize) -> Self {
        let total_elements = max_seq_len * hidden_size;

        let k_cache = CudaBuffer::new(total_elements);
        let v_cache = CudaBuffer::new(total_elements);

        KvCache {
            k_cache,
            v_cache,
            max_seq_len,
            hidden_size,
            current_pos: 0,
        }
    }

    /// Сброс кэша (вызывается, когда пользователь начинает новый диалог с чистого листа)
    pub fn clear(&mut self) {
        self.current_pos = 0;
    }

    /// Возвращает текущую длину накопленного контекста
    pub fn len(&self) -> usize {
        self.current_pos
    }

    /// Проверяет, не переполнился ли буфер контекста
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }

    /// Добавляет новые векторы Key и Value для одного токена в кэш видеокарты.
    /// Метод принимает объект как `&mut self`, так как внутренний счетчик позиции будет сдвигаться.
    pub fn append(&mut self, new_key: &CudaBuffer, new_value: &CudaBuffer) {
        if self.is_full() {
            panic!(
                "Ошибка: Превышен максимальный лимит контекста KV-Cache ({})!",
                self.max_seq_len
            );
        }

        let t_index = self.current_pos as i32;
        let h_size = self.hidden_size as i32;

        unsafe {
            launch_update_kv_cache(
                self.k_cache.as_raw_ptr(),
                self.v_cache.as_raw_ptr(),
                new_key.as_raw_ptr(),
                new_value.as_raw_ptr(),
                t_index,
                h_size,
            );
        }

        // Сдвигаем указатель контекста на следующий токен
        self.current_pos += 1;
    }
}
