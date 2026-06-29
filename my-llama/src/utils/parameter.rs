use crate::utils::CudaBuffer;

pub struct Parameter {
    pub data: CudaBuffer,
    pub grad: CudaBuffer,
    pub size: usize,
}

impl Parameter {
    pub fn new(elements: usize) -> Self {
        let data = CudaBuffer::new(elements);
        let grad = CudaBuffer::new(elements);

        grad.copy_from_host(&vec![0.0f32; elements]);

        Parameter {
            data,
            grad,
            size: elements,
        }
    }

    pub fn zero_grad(&self) {
        self.grad.copy_from_host(&vec![0.0f32; self.size]);
    }
}
