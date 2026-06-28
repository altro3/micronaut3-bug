from my_ai.utils.matrix_math import create_xavier_matrix, create_zero_matrix, matmul, transpose, softmax_row


class PureEmbedding:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = create_xavier_matrix(vocab_size, embedding_dim)
        self.grad_weights = create_zero_matrix(vocab_size, embedding_dim)
        self.last_input = None


    def forward(self, input_ids: list[int]) -> list[list[float]]:
        self.last_input = input_ids
        output = []
        for char_id in input_ids:
            output.append([num for num in self.weights[char_id]])
        return output

    def backward(self, grad_output: list[list[float]]) -> None:
        self.grad_weights = create_zero_matrix(self.vocab_size, self.embedding_dim)
        for position, char_id in enumerate(self.last_input):
            for i in range(self.embedding_dim):
                self.grad_weights[char_id][i] += grad_output[position][i]


class PureLinear:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = create_xavier_matrix(in_features, out_features)
        self.bias = [0.0 for _ in range(out_features)]
        self.grad_weights = create_zero_matrix(in_features, out_features)
        self.grad_bias = [0.0 for _ in range(out_features)]
        self.last_input = None

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        self.last_input = x
        base_out = matmul(x, self.weights)
        out = []
        for i in range(len(base_out)):
            new_row = [base_out[i][j] + self.bias[j] for j in range(self.out_features)]
            out.append(new_row)
        return out

    def backward(self, grad_output: list[list[float]]) -> list[list[float]]:
        self.grad_bias = [0.0 for _ in range(self.out_features)]
        for i in range(len(grad_output)):
            for j in range(self.out_features):
                self.grad_bias[j] += grad_output[i][j]

        X_T = transpose(self.last_input)
        self.grad_weights = matmul(X_T, grad_output)

        W_T = transpose(self.weights)
        grad_input = matmul(grad_output, W_T)
        return grad_input


class PureSoftmax:
    def __init__(self):
        self.last_output = None

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        out = [softmax_row(row) for row in x]
        self.last_output = out
        return out

    def backward(self, grad_output: list[list[float]]) -> list[list[float]]:
        grad_input = []
        for r in range(len(grad_output)):
            row_p = self.last_output[r]
            row_g = grad_output[r]
            n = len(row_p)
            row_in_grad = [0.0] * n
            for i in range(n):
                s = 0.0
                for j in range(n):
                    if i == j:
                        derivative = row_p[i] * (1.0 - row_p[j])
                    else:
                        derivative = -row_p[i] * row_p[j]
                    s += row_g[j] * derivative
                row_in_grad[i] = s
            grad_input.append(row_in_grad)
        return grad_input
