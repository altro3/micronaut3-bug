#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "windows")]
pub use windows::os::pin_thread;

#[cfg(target_os = "linux")]
pub unsafe fn pin_current_thread(_core_index: usize) {
    // Тут будет Linux код, а пока просто ничего не делаем
}