// matrix_math.c
#include <stddef.h>

#ifdef _WIN32
__declspec(dllexport) void c_matmul(const float* A, const float* B, float* res, int rows_A, int cols_A, int cols_B) {
#else
void c_matmul(const float* A, const float* B, float* res, int rows_A, int cols_A, int cols_B) {
#endif
    for (int i = 0; i < rows_A; i++) {
        int offset_A = i * cols_A;
        int offset_res = i * cols_B;
        for (int j = 0; j < cols_B; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                sum += A[offset_A + k] * B[k * cols_B + j];
            }
            res[offset_res + j] = sum;
        }
    }
}
