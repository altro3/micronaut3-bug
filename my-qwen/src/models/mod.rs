// 1. Говорим компилятору: "В этой папке есть файл layers.rs"
pub mod layers;
mod swiglu;

// 2. Делаем re-export (перепривязку), чтобы из main.rs можно было написать
//    "use models::RmsNorm", а не пробиваться через "models::layers::RmsNorm"
pub use layers::RmsNorm;
