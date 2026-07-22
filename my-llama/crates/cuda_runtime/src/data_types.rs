#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DataType {
    BF16 = 0,
    FP8 = 1,
    FP4 = 2,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum QuantType {
    None = 0,
    GgufQ4K = 1,
    GgufQ5K = 2,
    AwqInt4 = 3,
    GptqInt4 = 4,
    BlackwellFp4 = 5,
}
