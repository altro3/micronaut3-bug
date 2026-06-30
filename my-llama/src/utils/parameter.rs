use crate::utils::{CudaBuffer, CudaStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    Int4GPTQ,
    Int3AWQ,
}

impl DataType {
    pub fn element_size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::Int4GPTQ | DataType::Int3AWQ => 1, // Квантованные упакованные типы
        }
    }
}

pub struct Parameter {
    pub data: CudaBuffer,
    pub grad: Option<CudaBuffer>,
    pub m_buffer: Option<CudaBuffer>,
    pub v_buffer: Option<CudaBuffer>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub size: usize,
}

impl Parameter {
    pub fn new(
        shape: Vec<usize>,
        dtype: DataType,
        requires_grad: bool,
        stream: &CudaStream,
    ) -> Self {
        let size: usize = shape.iter().product();

        let bytes = match dtype {
            DataType::Int4GPTQ => (size + 7) / 8 * 4,
            DataType::Int3AWQ => (size * 3 + 7) / 8,
            _ => size * dtype.element_size(),
        };

        let data = CudaBuffer::new(bytes);

        let (grad, m_buffer, v_buffer) = if requires_grad {
            let grad_buffer = CudaBuffer::new(bytes);
            let m_buf = CudaBuffer::new(bytes);
            let v_buf = CudaBuffer::new(bytes);

            grad_buffer.zero_out_async(stream);
            m_buf.zero_out_async(stream);
            v_buf.zero_out_async(stream);

            (Some(grad_buffer), Some(m_buf), Some(v_buf))
        } else {
            (None, None, None)
        };

        Parameter {
            data,
            grad,
            m_buffer,
            v_buffer,
            shape,
            dtype,
            size,
        }
    }

    pub fn zero_grad_async(&self, stream: &CudaStream) {
        if let Some(ref grad_buffer) = self.grad {
            grad_buffer.zero_out_async(stream);
        }
    }

    pub fn load_weights_async<T: Copy>(&self, host_weights: &[T], stream: &CudaStream) {
        self.data.copy_from_host_async(host_weights, stream);
    }
}
