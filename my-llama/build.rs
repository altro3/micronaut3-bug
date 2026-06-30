fn main() {
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");

    let mut build = cc::Build::new();

    // Запускаем сборщик C++/CUDA кода
    build
        .cuda(true)
        .cudart("static")
        .flag("-arch=sm_90")
        .file("cuda/src/matmul.cu")
        .file("cuda/src/swiglu.cu")
        .file("cuda/src/linear_backward.cu")
        .file("cuda/src/argmax.cu")
        .file("cuda/src/rmsnorm.cu")
        .file("cuda/src/softmax.cu")
        .file("cuda/src/attention.cu")
        .file("cuda/src/adam.cu")
        .include("cuda/include");

    build.compile("cuda_kernels");

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
