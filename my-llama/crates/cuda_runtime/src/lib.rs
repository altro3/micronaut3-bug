use std::ffi::c_void;
use std::ptr;

pub mod input_stage;

unsafe extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetLastError() -> i32;
    pub fn cudaStreamCreateWithFlags(p_stream: *mut *mut c_void, flags: u32) -> i32;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    pub fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    pub fn cudaEventDestroy(event: *mut c_void) -> i32;
    pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    pub fn cudaEventSynchronize(event: *mut c_void) -> i32;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    pub fn cudaDeviceSetLimit(limit: i32, value: usize) -> i32;
}

pub struct CudaBuffer {
    pub ptr: *mut c_void,
    pub size_bytes: usize,
}

impl CudaBuffer {
    pub fn alloc(size_bytes: usize) -> Self {
        let mut raw_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            let res = cudaMalloc(&mut raw_ptr, size_bytes);
            assert_eq!(res, 0);
        }
        CudaBuffer { ptr: raw_ptr, size_bytes }
    }

    /// # Safety
    /// `host_data` must be a valid pointer to initialized memory of at least `bytes` size.
    pub unsafe fn copy_to_device(&self, host_data: *const c_void, bytes: usize) {
        assert!(bytes <= self.size_bytes);
        unsafe {
            cudaMemcpy(self.ptr, host_data, bytes, 1);
        }
    }

    /// # Safety
    /// `host_data` must be a valid pointer to memory capable of holding at least `bytes` data.
    pub unsafe fn copy_to_host(&self, host_data: *mut c_void, bytes: usize) {
        assert!(bytes <= self.size_bytes);
        unsafe {
            cudaMemcpy(host_data, self.ptr as *const c_void, bytes, 2);
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

pub fn device_synchronize() -> i32 {
    unsafe { cudaDeviceSynchronize() }
}

pub fn get_last_error() -> i32 {
    unsafe { cudaGetLastError() }
}

pub fn set_device_limit(limit: i32, value: usize) -> i32 {
    unsafe { cudaDeviceSetLimit(limit, value) }
}

pub fn stream_create_with_flags(flags: u32) -> *mut c_void {
    let mut stream: *mut c_void = ptr::null_mut();
    unsafe {
        cudaStreamCreateWithFlags(&mut stream, flags);
    }
    stream
}

/// # Safety
/// `stream` must be a valid initialized CUDA stream pointer.
pub unsafe fn stream_destroy(stream: *mut c_void) {
    unsafe {
        cudaStreamDestroy(stream);
    }
}

pub fn event_create() -> *mut c_void {
    let mut event: *mut c_void = ptr::null_mut();
    unsafe {
        cudaEventCreate(&mut event);
    }
    event
}

/// # Safety
/// `event` must be a valid initialized CUDA event pointer.
pub unsafe fn event_destroy(event: *mut c_void) {
    unsafe {
        cudaEventDestroy(event);
    }
}

/// # Safety
/// Both `event` and `stream` must be valid initialized CUDA runtime pointers.
pub unsafe fn event_record(event: *mut c_void, stream: *mut c_void) {
    unsafe {
        cudaEventRecord(event, stream);
    }
}

/// # Safety
/// `event` must be a valid initialized CUDA event pointer.
pub unsafe fn event_synchronize(event: *mut c_void) {
    unsafe {
        cudaEventSynchronize(event);
    }
}

/// # Safety
/// Both `start` and `end` must be valid initialized CUDA event pointers.
pub unsafe fn event_elapsed_time(start: *mut c_void, end: *mut c_void) -> f32 {
    let mut ms = 0.0_f32;
    unsafe {
        cudaEventElapsedTime(&mut ms, start, end);
    }
    ms
}
