use std::path::PathBuf;
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");

    let cuda_path = env::var("CUDA_PATH")
        .expect("Критическая ошибка: Переменная среды CUDA_PATH не найдена! Проверь установку CUDA Toolkit.");

    let cuda_lib_dir = PathBuf::from(&cuda_path).join("lib").join("x64");
    if !cuda_lib_dir.exists() {
        panic!("Критическая ошибка: Директория библиотек CUDA не существует по пути: {}", cuda_lib_dir.display());
    }

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    let mut build = cc::Build::new();

    build
        .cuda(true)
        .cudart("static")
        .flag("-gencode=arch=compute_100,code=sm_100")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-O3")
        .flag("--use_fast_math")
        .include("cuda/include");

    let paths = fs::read_dir("cuda/src")
        .expect("Критическая ошибка: Не удалось прочитать папку cuda/src")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "cu"));

    for path in paths {
        build.file(path);
    }

    build.compile("cuda_kernels");

    println!("cargo:rustc-link-lib=static=cuda_kernels");
    // println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
