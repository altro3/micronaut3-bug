import math

from my_ai.models.pure_layers import PureEmbedding, PureLinear, PureSoftmax
from my_ai.utils.matrix_math import matmul, transpose, create_zero_matrix


class PureSelfAttention:
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size

        self.query_layer = PureLinear(n_embd, head_size)
        self.key_layer = PureLinear(n_embd, head_size)
        self.value_layer = PureLinear(n_embd, head_size)
        self.softmax = PureSoftmax()

        self.last_x = None
        self.last_q = None
        self.last_k = None
        self.last_v = None
        self.last_wei = None

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        T = len(x)
        self.last_x = x

        self.last_q = self.query_layer.forward(x)
        self.last_k = self.key_layer.forward(x)
        self.last_v = self.value_layer.forward(x)

        k_T = transpose(self.last_k)
        wei = matmul(self.last_q, k_T)

        scale = 1.0 / math.sqrt(self.head_size)
        for i in range(T):
            for j in range(T):
                wei[i][j] *= scale

        for i in range(T):
            for j in range(T):
                if j > i:
                    wei[i][j] = float('-inf')

        self.last_wei = self.softmax.forward(wei)
        out = matmul(self.last_wei, self.last_v)
        return out

    def backward(self, grad_output: list[list[float]]) -> list[list[float]]:
        T = len(grad_output)
        scale = 1.0 / math.sqrt(self.head_size)

        wei_T = transpose(self.last_wei)
        grad_v = matmul(wei_T, grad_output)

        v_T = transpose(self.last_v)
        dwei_from_v = matmul(grad_output, v_T)

        for i in range(T):
            for j in range(T):
                if j > i:
                    dwei_from_v[i][j] = 0.0

        grad_wei = create_zero_matrix(T, T)
        for r in range(T):
            row_p = self.last_wei[r]
            row_g = dwei_from_v[r]

            dot_product = sum(g * p for g, p in zip(row_g, row_p))

            for i in range(T):
                grad_wei[r][i] = row_p[i] * (row_g[i] - dot_product) * scale

        grad_q = matmul(grad_wei, self.last_k)
        grad_wei_T = transpose(grad_wei)
        grad_k = matmul(grad_wei_T, self.last_q)

        grad_x_from_q = self.query_layer.backward(grad_q)
        grad_x_from_k = self.key_layer.backward(grad_k)
        grad_x_from_v = self.value_layer.backward(grad_v)

        grad_x = create_zero_matrix(T, self.n_embd)
        for i in range(T):
            for j in range(self.n_embd):
                grad_x[i][j] = grad_x_from_q[i][j] + grad_x_from_k[i][j] + grad_x_from_v[i][j]

        return grad_x


class PureTransformerLM:
    def __init__(self, vocab_size: int, block_size: int, n_embd: int = 32):
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embeddings = PureEmbedding(vocab_size, n_embd)
        self.position_embeddings = PureEmbedding(block_size, n_embd)

        self.attention = PureSelfAttention(n_embd, head_size=n_embd, block_size=block_size)

        from my_ai.models.pure_layers import PureReLU, PureLinear
        self.mlp_fc1 = PureLinear(n_embd, n_embd * 2)
        self.mlp_relu = PureReLU()
        self.mlp_fc2 = PureLinear(n_embd * 2, n_embd)

        self.lm_head = PureLinear(n_embd, vocab_size)

    def zero_grad(self) -> None:
        self.token_embeddings.zero_grad()
        self.position_embeddings.zero_grad()
        self.attention.query_layer.zero_grad()
        self.attention.key_layer.zero_grad()
        self.attention.value_layer.zero_grad()
        self.mlp_fc1.zero_grad()
        self.mlp_fc2.zero_grad()
        self.lm_head.zero_grad()

    def forward(self, input_ids: list) -> list[list[float]]:
        if input_ids and isinstance(input_ids[0], list):
            B = len(input_ids)
            T = len(input_ids[0])

            tok_emb = []
            pos_emb = []
            for b in range(B):
                tok_emb.extend(self.token_embeddings.forward(input_ids[b]))
                pos_emb.extend(self.position_embeddings.forward(list(range(T))))
            total_tokens = B * T
        else:
            T = len(input_ids)
            tok_emb = self.token_embeddings.forward(input_ids)
            pos_emb = self.position_embeddings.forward(list(range(T)))
            total_tokens = T

        x = create_zero_matrix(total_tokens, self.token_embeddings.embedding_dim)
        for i in range(total_tokens):
            for j in range(self.token_embeddings.embedding_dim):
                x[i][j] = tok_emb[i][j] + pos_emb[i][j]

        if input_ids and isinstance(input_ids[0], list):
            B = len(input_ids)
            T = len(input_ids[0])
            attention_out = []
            for b in range(B):
                sub_x = x[b * T: (b + 1) * T]
                attention_out.extend(self.attention.forward(sub_x))
            x = attention_out
        else:
            x = self.attention.forward(x)

        x = self.mlp_fc1.forward(x)
        x = self.mlp_relu.forward(x)
        x = self.mlp_fc2.forward(x)

        logits = self.lm_head.forward(x)
        return logits

    def backward(self, grad_output: list[list[float]]) -> None:
        grad_x = self.lm_head.backward(grad_output)

        grad_x = self.mlp_fc2.backward(grad_x)
        grad_x = self.mlp_relu.backward(grad_x)
        grad_x = self.mlp_fc1.backward(grad_x)

        if hasattr(self, 'last_input_is_batch') and self.last_input_is_batch:
            pass

        grad_x = self.attention.backward(grad_x)
        self.token_embeddings.backward(grad_x)
        self.position_embeddings.backward(grad_x)

    def get_parameters(self) -> list[tuple[list[list[float]], list[list[float]]]]:
        return [
            (self.token_embeddings.weights, self.token_embeddings.grad_weights),
            (self.position_embeddings.weights, self.position_embeddings.grad_weights),
            (self.attention.query_layer.weights, self.attention.query_layer.grad_weights),
            (self.attention.key_layer.weights, self.attention.key_layer.grad_weights),
            (self.attention.value_layer.weights, self.attention.value_layer.grad_weights),
            (self.mlp_fc1.weights, self.mlp_fc1.grad_weights),
            (self.mlp_fc2.weights, self.mlp_fc2.grad_weights),
            (self.lm_head.weights, self.lm_head.grad_weights)
        ]

    def get_biases(self) -> list[tuple[list[float], list[float]]]:
        return [
            (self.attention.query_layer.bias, self.attention.query_layer.grad_bias),
            (self.attention.key_layer.bias, self.attention.key_layer.grad_bias),
            (self.attention.value_layer.bias, self.attention.value_layer.grad_bias),
            (self.mlp_fc1.bias, self.mlp_fc1.grad_bias),
            (self.mlp_fc2.bias, self.mlp_fc2.grad_bias),
            (self.lm_head.bias, self.lm_head.grad_bias)
        ]
