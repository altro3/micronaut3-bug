#[cfg(target_os = "windows")]
pub mod os {
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
        fn SetThreadPriority(thread: *mut std::ffi::c_void, priority: i32) -> i32;
    }

    pub fn pin_thread(core_index: usize) {
        unsafe {
            let thread_handle = GetCurrentThread();
            let mask = 1 << core_index;
            SetThreadAffinityMask(thread_handle, mask);
            SetThreadPriority(thread_handle, 2);
        }
    }
}
