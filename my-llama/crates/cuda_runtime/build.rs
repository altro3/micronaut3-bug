use std::path::PathBuf;

fn main() {
println!("cargo:rerun-if-changed=../../backends/cuda/src");
println!("cargo:rerun-if-changed=../../backends/cuda/include");
println!("cargo:rerun-if-changed=../../backends/cuda/CMakeLists.txt");

let cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3";
let nvcc_path = format!(r"{}\bin\nvcc.exe", cuda_path);
let ninja_path = r"C:\Users\alexu\scoop\apps\ninja\current\ninja.exe";

let dst = cmake::Config::new("../../backends/cuda")
.generator("Ninja")
.profile("Release")
.no_build_target(true)
.define("CMAKE_CUDA_COMPILER", &nvcc_path)
.define("CMAKE_MAKE_PROGRAM", &ninja_path)
.define("CMAKE_CUDA_STANDARD", "20")
.define("CMAKE_CUDA_STANDARD_REQUIRED", "ON")
.define("CMAKE_CUDA_FLAGS", "-Xcompiler /Zc:preprocessor")
.define("CMAKE_CXX_FLAGS", "/Zc:preprocessor")
.define("CMAKE_NETRC", "OPTIONAL")
.define("FETCH_CUTLASS", "ON")
.define("FETCH_FLASHINFER", "ON")
.define("ISOLATED_BUILD", "OFF")
.define("RUST_BACKTRACE", "full")
.build();


let lib_dir = dst.join("build").join("src");
println!("cargo:rustc-link-search=native={}", lib_dir.display());

let cuda_lib_dir = PathBuf::from(cuda_path).join("lib").join("x64");
println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

println!("cargo:rustc-link-lib=static=cuda_kernels");
println!("cargo:rustc-link-lib=static=cudart_static");
println!("cargo:rustc-link-lib=dylib=cublas");
println!("cargo:rustc-link-lib=dylib=cublasLt");
println!("cargo:rustc-link-lib=dylib=advapi32");
println!("cargo:rustc-link-lib=dylib=user32");
}
