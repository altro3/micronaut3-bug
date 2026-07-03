use std::arch::x86_64::*;

pub const HASH_SIZE: usize = 524288; // 2^19
pub const HASH_MASK: usize = HASH_SIZE - 1;

const CTRL_OFFSET: usize = 0;
const KEYS_OFFSET: usize = HASH_SIZE;
const VALS_OFFSET: usize = HASH_SIZE + HASH_SIZE * 8;
const TOTAL_MEMORY_SIZE: usize = HASH_SIZE + (HASH_SIZE * 8) + (HASH_SIZE * 4);

pub struct InlineHashTable {
    ptr: *mut u8,
}

impl InlineHashTable {
    pub fn new() -> Self {
        let layout = std::alloc::Layout::from_size_align(TOTAL_MEMORY_SIZE, 64).unwrap();
        let memory = unsafe {
            let p = std::alloc::alloc(layout);
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            p
        };

        let mut s = Self { ptr: memory };

        unsafe {
            std::ptr::write_bytes(s.control_mut_ptr(), 0x80, HASH_SIZE);
            std::ptr::write_bytes(s.keys_mut_ptr(), 0, HASH_SIZE);

            let vals_ptr = s.vals_mut_ptr();
            for i in 0..HASH_SIZE {
                *vals_ptr.add(i) = u32::MAX;
            }
        }
        s
    }

    #[inline(always)]
    fn control_ptr(&self) -> *const u8 {
        unsafe { self.ptr.add(CTRL_OFFSET) }
    }
    #[inline(always)]
    fn control_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.ptr.add(CTRL_OFFSET) }
    }

    #[inline(always)]
    fn keys_ptr(&self) -> *const u64 {
        unsafe { self.ptr.add(KEYS_OFFSET) as *const u64 }
    }
    #[inline(always)]
    fn keys_mut_ptr(&mut self) -> *mut u64 {
        unsafe { self.ptr.add(KEYS_OFFSET) as *mut u64 }
    }

    #[inline(always)]
    fn vals_ptr(&self) -> *const u32 {
        unsafe { self.ptr.add(VALS_OFFSET) as *const u32 }
    }
    #[inline(always)]
    fn vals_mut_ptr(&mut self) -> *mut u32 {
        unsafe { self.ptr.add(VALS_OFFSET) as *mut u32 }
    }

    #[inline(always)]
    pub fn insert(&mut self, hash: u64, id: u32) {
        let h2 = (hash & 0x7F) as u8;
        let mut idx = (hash as usize) & HASH_MASK;

        unsafe {
            let ctrl_base = self.control_mut_ptr();
            let keys_base = self.keys_mut_ptr();
            let vals_base = self.vals_mut_ptr();

            loop {
                let ctrl = *ctrl_base.add(idx);
                if ctrl == 0x80 {
                    *ctrl_base.add(idx) = h2;
                    *keys_base.add(idx) = hash;
                    *vals_base.add(idx) = id;
                    return;
                }
                idx = (idx + 1) & HASH_MASK;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn find_avx2(&self, hash: u64) -> u32 {
        let mut base_idx = (hash as usize) & HASH_MASK;
        let h2 = (hash & 0x7F) as u8;

        let target_ctrls = _mm256_set1_epi8(h2 as i8);
        let empty_ctrls = _mm256_set1_epi8(0x80u8 as i8);

        let ctrl_base = self.control_ptr();
        let keys_base = self.keys_ptr();
        let vals_base = self.vals_ptr();

        loop {
            let current_group_start = base_idx & !31;
            let ctrl_ptr = unsafe { ctrl_base.add(current_group_start) };
            let group_offset = base_idx & 31;

            let group = unsafe { _mm256_loadu_si256(ctrl_ptr as *const __m256i) };

            let cmp_match = _mm256_cmpeq_epi8(group, target_ctrls);
            let mut match_mask = _mm256_movemask_epi8(cmp_match) as u32;

            match_mask &= !((1u32 << group_offset) - 1);

            while match_mask != 0 {
                let next_bit = match_mask.trailing_zeros() as usize;
                let actual_idx = current_group_start + next_bit;

                unsafe {
                    if *keys_base.add(actual_idx) == hash {
                        return *vals_base.add(actual_idx);
                    }
                }

                match_mask &= match_mask - 1;
            }

            let cmp_empty = _mm256_cmpeq_epi8(group, empty_ctrls);
            let empty_mask = _mm256_movemask_epi8(cmp_empty) as u32;

            if empty_mask != 0 {
                let first_empty_bit = empty_mask.trailing_zeros() as usize;

                if first_empty_bit >= group_offset {
                    return u32::MAX;
                }

                return u32::MAX;
            }

            base_idx = (current_group_start.wrapping_add(32)) & HASH_MASK;
        }
    }

    #[inline(always)]
    pub fn find(&self, hash: u64) -> u32 {
        #[cfg(target_feature = "avx2")]
        unsafe {
            return self.find_avx2(hash);
        }

        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            let mut idx = (hash as usize) & HASH_MASK;
            let h2 = (hash & 0x7F) as u8;
            let ctrl_base = self.control_ptr();
            let keys_base = self.keys_ptr();
            let vals_base = self.vals_ptr();
            loop {
                let ctrl = *ctrl_base.add(idx);
                if ctrl == h2 && *keys_base.add(idx) == hash {
                    return *vals_base.add(idx);
                }
                if ctrl == 0x80 {
                    return u32::MAX;
                }
                idx = (idx + 1) & HASH_MASK;
            }
        }
    }
}

impl Drop for InlineHashTable {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(TOTAL_MEMORY_SIZE, 64).unwrap();
        unsafe {
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}

unsafe impl Send for InlineHashTable {}
unsafe impl Sync for InlineHashTable {}
