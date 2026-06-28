import math
import random


def create_xavier_matrix(rows: int, cols: int) -> list[list[float]]:
    bound = 1.0 / math.sqrt(rows) if rows > 0 else 0.1
    return [[random.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)]


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
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    result = [[0.0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    return result


def softmax_row(row: list[float]) -> list[float]:
    if not row: return []
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    sum_exps = sum(exps)
    if sum_exps == 0: sum_exps = 1e-15
    return [e / sum_exps for e in exps]
