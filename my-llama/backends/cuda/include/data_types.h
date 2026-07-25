#ifndef DATA_TYPES_H
#define DATA_TYPES_H

enum class DataType {
    BF16 = 0,
    FP8 = 1,
    FP4 = 2
};

enum class QuantType {
    NONE = 0,
    GGUF_Q4_K = 1,
    GGUF_Q5_K = 2,
    AWQ_INT4 = 3,
    GPTQ_INT4 = 4,
    BLACKWELL_FP4 = 5
};
#endif
