import math
import random
from array import array

Matrix = tuple[array, int, int]


def create_xavier_matrix(inputs: int, outputs: int) -> list[list[float]]:
    if inputs == 0: return [[0.0 for _ in range(outputs)]]
    bound = math.sqrt(6.0 / (inputs + outputs))
    return [[random.uniform(-bound, bound) for _ in range(outputs)] for _ in range(inputs)]


def create_zero_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    if not A or not B or not B:
        return []

    rows_A = len(A)

    B_T = [list(x) for x in zip(*B)]
    cols_B = len(B_T)

    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        row_A = A[i]
        for j in range(cols_B):
            col_B = B_T[j]
            result[i][j] = sum(a * b for a, b in zip(row_A, col_B))

    return result


def transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(x) for x in zip(*A)]

def softmax_row(row: list[float]) -> list[float]:
    if not row: return []
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    sum_exps = sum(exps)
    if sum_exps == 0: sum_exps = 1e-15
    return [e / sum_exps for e in exps]
