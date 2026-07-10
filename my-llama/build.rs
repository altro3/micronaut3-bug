use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");

    let cuda_path = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3");

    let cuda_lib_dir = cuda_path.join("lib").join("x64");
    let cuda_bin_dir = cuda_path.join("bin");

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    let nvcc_path = cuda_bin_dir.join("nvcc.exe");
    unsafe {
        env::set_var("NVCC", nvcc_path);
    }

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-gencode=arch=compute_120,code=sm_120")
        .flag("-gencode=arch=compute_120,code=compute_120")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-O3")
        .flag("-std=c++20")
        .flag("--use_fast_math")
        .include("cuda/include");

    let target = env::var("TARGET").unwrap();
    if target.contains("msvc") {
        build.flag("-Xcompiler").flag("/EHsc");
    }

    let paths = fs::read_dir("cuda/src")
        .expect("Не удалось прочитать директорию cuda/src")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "cu"));

    for path in paths {
        build.file(path);
    }

    build.compile("cuda_kernels");
}
