use crate::utils::buffer::{CudaBuffer, CudaStream};

pub struct Parameter {
    pub data: CudaBuffer,
    pub grad: CudaBuffer,
    pub size: usize,
}

impl Parameter {
    pub fn new(elements: usize, stream: &CudaStream) -> Self {
        let data = CudaBuffer::new(elements);
        let grad = CudaBuffer::new(elements);

        grad.zero_out_async(stream);

        Parameter {
            data,
            grad,
            size: elements,
        }
    }

    pub fn zero_grad_async(&self, stream: &CudaStream) {
        self.grad.zero_out_async(stream);
    }

    pub fn load_weights_async<T: Copy>(&self, host_weights: &[T], stream: &CudaStream) {
        self.data.copy_from_host_async(host_weights, stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_gradient_zeroing_async() {
        const ELEMENTS: usize = 5;
        let stream = CudaStream::new();

        let param = Parameter::new(ELEMENTS, &stream);

        let mut check_grads = vec![0.0f32; ELEMENTS];
        param.grad.copy_to_host_async(&mut check_grads, &stream);

        stream.synchronize();
        assert_eq!(
            check_grads,
            vec![0.0f32; ELEMENTS],
            "Градиенты не занулены при старте"
        );

        let fake_gradients = vec![0.5, -1.2, 3.14, 0.0, 99.9];
        param.grad.copy_from_host_async(&fake_gradients, &stream);

        param.zero_grad_async(&stream);

        param.grad.copy_to_host_async(&mut check_grads, &stream);
        stream.synchronize();

        assert_eq!(
            check_grads,
            vec![0.0f32; ELEMENTS],
            "Метод zero_grad_async не очистил память на GPU! Получено: {:?}",
            check_grads
        );
    }
}
