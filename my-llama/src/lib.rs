pub mod models;
pub mod token;
pub mod utils;

unsafe extern "C" {
    fn init_cublas_infrastructure();
}

pub fn init_framework() {
    unsafe {
        init_cublas_infrastructure();
    }
    println!(
        "[MY-LLAMA] Высокопроизводительный движок на базе асинхронной архитектуры успешно запущен."
    );
}
