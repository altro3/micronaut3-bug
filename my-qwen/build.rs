fn main() {
    // Говорим Cargo: если файлы в папке cuda изменятся, нужно перезапустить этот скрипт сборки
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");

    // Запускаем сборщик C++/CUDA кода
    cc::Build::new()
        .cuda(true)          // Включаем поддержку CUDA (cc сам начнет искать утилиту nvcc)
        .cudart("static")    // Статически вшиваем CUDA Runtime в нашу программу
        .flag("-arch=sm_90") // Указываем архитектуру Blackwell для вашей RTX 5090
        .file("cuda/src/matmul.cu") // Файлы, которые нужно скомпилировать
        .file("cuda/src/softmax.cu")
        .file("cuda/src/adam.cu")
        .include("cuda/include")    // Где искать заголовочные файлы (.h)
        .compile("cuda_kernels");   // Имя итоговой библиотеки, которую мы получим

    // Помогаем линкеру Windows найти системную библиотеку cuda.lib
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
