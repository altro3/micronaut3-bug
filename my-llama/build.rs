use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");

    let cuda_path = env::var("CUDA_PATH").expect("Fatal error: CUDA_PATH environment variable not found!");
    let cuda_lib_dir = PathBuf::from(&cuda_path).join("lib").join("x64");
    let nvcc_path = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    unsafe { env::set_var("NVCC", nvcc_path); }

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cudart("shared")
        .flag("-gencode=arch=compute_100,code=sm_100")
        .flag("-gencode=arch=compute_90,code=sm_90")
        .flag("-gencode=arch=compute_89,code=sm_89")
        .flag("-O3")
        .flag("--use_fast_math")
        .include("cuda/include");

    let target = env::var("TARGET").unwrap();
    if target.contains("msvc") {
        build.flag("-Xcompiler").flag("/EHsc");
    }

    let paths = fs::read_dir("cuda/src")
        .expect("Failed to read cuda/src directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "cu"));

    for path in paths {
        build.file(path);
    }

    build.compile("cuda_kernels");
}
