use std::path::PathBuf;

fn main() {
    // Cargo будет перезапускать этот скрипт только при изменении исходников CUDA в новой папке backends
    println!("cargo:rerun-if-changed=backends/cuda/src");
    println!("cargo:rerun-if-changed=backends/cuda/include");

    let cuda_path = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3");
    let cuda_lib_dir = cuda_path.join("lib").join("x64");

    // Пути поиска для линковщика Rust (системные динамические либы NVIDIA)
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    // Забираем нашу скомпилированную батником статическую библиотеку ядер (те самые 2.5 МБ)
    let compiled_lib_dir = PathBuf::from(r"D:\.b_llama\backends\cuda\Release");

    println!("cargo:rustc-link-search=native={}", compiled_lib_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_kernels");
}
