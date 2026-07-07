use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");

    let mut build = cc::Build::new();

    build.cuda(true).cudart("static").flag("-arch=sm_90").flag("-O3").include("cuda/include");

    let paths = fs::read_dir("cuda/src")
        .expect("Критическая ошибка: Не удалось прочитать папку cuda/src")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "cu"));

    for path in paths {
        build.file(path);
    }

    build.compile("cuda_kernels");

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    }

    build.compile("cuda_kernels");

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    }

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
