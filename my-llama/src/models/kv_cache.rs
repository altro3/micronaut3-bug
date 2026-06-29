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

pub struct KvCache {
    pub k_cache: CudaBuffer,
    pub v_cache: CudaBuffer,
    max_seq_len: usize,
    hidden_size: usize,
    current_pos: usize,
}

impl KvCache {
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

    pub fn clear(&mut self) {
        self.current_pos = 0;
    }

    pub fn len(&self) -> usize {
        self.current_pos
    }

    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }

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

        self.current_pos += 1;
    }
}
