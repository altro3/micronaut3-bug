use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../backends/cuda/src");
    println!("cargo:rerun-if-changed=../../backends/cuda/include");

    let bat_path = Path::new("..\\..\\build_cuda.bat");
    let status = Command::new("cmd")
        .args(&["/C", bat_path.to_str().unwrap()])
        .status()
        .expect("Не удалось запустить build_cuda.bat! Проверь путь к файлу.");
    if !status.success() {
        panic!("🚨 ОШИБКА СБОРКИ CUDA: Скрипт build_cuda.bat завершился с ненулевым кодом!");
    }

    let cuda_path = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3");
    let cuda_lib_dir = cuda_path.join("lib").join("x64");
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    let compiled_lib_dir = PathBuf::from(r"D:\.my_llama\backends\cuda\src\Release");
    println!("cargo:rustc-link-search=native={}", compiled_lib_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=advapi32");
}
