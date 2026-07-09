use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

#[link(name = "cuda_kernels", kind = "static")]
#[link(name = "cudart", kind = "dylib")]
#[link(name = "cublas", kind = "dylib")]
#[link(name = "cublasLt", kind = "dylib")]
unsafe extern "C" {
    pub fn launch_adamw(
        w: *mut f32,
        g: *mut f32,
        m: *mut f32,
        v: *mut f32,
        sz: i32,
        lr: f32,
        b1: f32,
        b2: f32,
        e: f32,
        wd: f32,
        st: f32,
        s: *mut c_void,
    );
}

struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
}

impl CudaBuffer {
    fn alloc(size: usize) -> Self {
        let mut ptr = ptr::null_mut();
        unsafe {
            unsafe extern "C" {
                fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
            }
            let res = cudaMalloc(&mut ptr as *mut *mut f32 as *mut *mut c_void, size * 4);
            assert_eq!(res, 0, "Ошибка: не удалось выделить память на GPU");
        }
        CudaBuffer { ptr, size }
    }

    fn copy_to_device(&self, host_data: &[f32]) {
        unsafe {
            unsafe extern "C" {
                fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
            }
            cudaMemcpy(self.ptr as *mut c_void, host_data.as_ptr() as *const c_void, self.size * 4, 1);
        }
    }

    fn copy_to_host(&self, host_data: &mut [f32]) {
        unsafe {
            unsafe extern "C" {
                fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
            }
            cudaMemcpy(host_data.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, self.size * 4, 2);
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            unsafe extern "C" {
                fn cudaFree(dev_ptr: *mut c_void) -> i32;
            }
            cudaFree(self.ptr as *mut c_void);
        }
    }
}

fn main() {
    println!("=== ТЕСТИРОВАНИЕ СКОРОСТИ И ТОЧНОСТИ ЯДРА ADAMW ===");

    let size = 26_255_376;
    let mem_bytes = size * 4 * 4;
    println!(
        "Размер тестового тензора: {} элементов (~{:.2} МБ общей памяти GPU)",
        size,
        mem_bytes as f64 / 1024.0 / 1024.0
    );

    let lr = 1e-4;
    let beta1 = 0.9;
    let beta2 = 0.95;
    let epsilon = 1e-8;
    let weight_decay = 0.01;
    let step = 12.0;

    let h_w = vec![0.5f32; size];
    let h_g = vec![0.02f32; size];
    let h_m = vec![0.005f32; size];
    let h_v = vec![0.0001f32; size];

    let d_w = CudaBuffer::alloc(size);
    let d_g = CudaBuffer::alloc(size);
    let d_m = CudaBuffer::alloc(size);
    let d_v = CudaBuffer::alloc(size);

    d_w.copy_to_device(&h_w);
    d_g.copy_to_device(&h_g);
    d_m.copy_to_device(&h_m);
    d_v.copy_to_device(&h_v);

    println!("Запуск ядра на RTX 5090...");

    unsafe {
        launch_adamw(
            d_w.ptr,
            d_g.ptr,
            d_m.ptr,
            d_v.ptr,
            size as i32,
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step,
            ptr::null_mut(),
        );
    }

    d_w.copy_to_device(&h_w);
    d_g.copy_to_device(&h_g);

    let start = Instant::now();
    unsafe {
        launch_adamw(
            d_w.ptr,
            d_g.ptr,
            d_m.ptr,
            d_v.ptr,
            size as i32,
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step,
            ptr::null_mut(),
        );
        unsafe extern "C" {
            fn cudaDeviceSynchronize() -> i32;
        }
        cudaDeviceSynchronize();
    }
    let duration = start.elapsed();

    let bytes_processed = size * 4 * 7;
    let bandwidth_gbps = (bytes_processed as f64 / 1e9) / duration.as_secs_f64();

    println!("Время выполнения на GPU: {:?}", duration);
    println!("Вычисленная пропускная способность VRAM: {:.2} ГБ/сек", bandwidth_gbps);

    let mut final_w = vec![0.0f32; size];
    let mut final_g = vec![0.0f32; size];
    d_w.copy_to_host(&mut final_w);
    d_g.copy_to_host(&mut final_g);

    let bc1 = 1.0 - beta1.powf(step);
    let bc2 = 1.0 - beta2.powf(step);
    let m_expected = beta1 * h_m[0] + (1.0 - beta1) * h_g[0];
    let v_expected = beta2 * h_v[0] + (1.0 - beta2) * h_g[0] * h_g[0];
    let m_hat = m_expected / bc1;
    let v_hat = v_expected / bc2;
    let w_expected = h_w[0] - lr * ((m_hat / (v_hat.sqrt() + epsilon)) + weight_decay * h_w[0]);

    let error = (final_w[0] - w_expected).abs();
    println!("\nПроверка точности первого элемента:");
    println!("Ожидалось (CPU): {}", w_expected);
    println!("Получено (GPU):  {}", final_w[0]);
    println!("Абсолютная ошибка математики: {:e}", error);
    println!("Градиент обнулен в памяти: {}", final_g[0] == 0.0f32);

    if error < 1e-5 && final_g[0] == 0.0 {
        println!("🚀 СУПЕР! Ядро AdamW работает идеально точно.");
    } else {
        println!("❌ ОШИБКА! Математика расходится, проверяй формулу.");
    }
}
