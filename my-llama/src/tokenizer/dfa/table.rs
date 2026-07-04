use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

pub struct DfaTransitionTable {
    ptr: *mut u16,
    num_states: usize,
    layout: Layout,
}

impl DfaTransitionTable {
    pub fn new(num_states: usize) -> Self {
        let size = num_states * 256 * size_of::<u16>();
        let layout = Layout::from_size_align(size, 64).unwrap();

        let ptr = unsafe {
            let p = alloc(layout) as *mut u16;
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            ptr::write_bytes(p, 0xFF, num_states * 256);
            p
        };

        Self { ptr, num_states, layout }
    }

    #[inline(always)]
    pub fn set_transition(&mut self, from_state: u16, byte: u8, to_state: u16) {
        if (from_state as usize) < self.num_states {
            unsafe {
                let idx = ((from_state as usize) << 8) | (byte as usize);
                *self.ptr.add(idx) = to_state;
            }
        }
    }

    #[inline(always)]
    pub unsafe fn get_next_state(&self, current_state: u16, byte: u8) -> u16 {
        let idx = ((current_state as usize) << 8) | (byte as usize);
        unsafe { *self.ptr.add(idx) }
    }

    #[inline(always)]
    pub unsafe fn ptr_mut(&mut self) -> *mut u16 {
        self.ptr
    }

    pub fn num_states(&self) -> usize {
        self.num_states
    }
}

impl Drop for DfaTransitionTable {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

unsafe impl Send for DfaTransitionTable {}
unsafe impl Sync for DfaTransitionTable {}
