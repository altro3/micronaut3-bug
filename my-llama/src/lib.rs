pub mod cuda;
pub mod models;
pub mod sys;
pub mod tokenizer;
pub mod utils;
pub mod test_utils;

// unsafe extern "C" {
//     pub fn init_cublas_infrastructure();
// }

pub fn init_framework() {
    // unsafe {
    //     init_cublas_infrastructure();
    // }
    println!("[MY-LLAMA] Высокопроизводительный движок на базе асинхронной архитектуры арен успешно запущен.");
}
