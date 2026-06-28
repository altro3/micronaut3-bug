import math
import random


def create_xavier_matrix(inputs: int, outputs: int) -> list[list[float]]:
    if inputs == 0: return [[0.0 for _ in range(outputs)]]
    bound = math.sqrt(6.0 / (inputs + outputs))
    return [[random.uniform(-bound, bound) for _ in range(outputs)] for _ in range(inputs)]


def create_zero_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    cols_B = len(B[0]) if len(B) > 0 else 0

    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            s = 0.0
            for k in range(cols_A):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def transpose(A: list[list[float]]) -> list[list[float]]:
    if not A or not A[0]: return []
    return [list(item) for item in zip(*A)]


def softmax_row(row: list[float]) -> list[float]:
    if not row: return []
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    sum_exps = sum(exps)
    if sum_exps == 0: sum_exps = 1e-15
    return [e / sum_exps for e in exps]


def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0
