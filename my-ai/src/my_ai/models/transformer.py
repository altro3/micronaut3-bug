import torch
import torch.nn as nn
from torch.nn import functional as F

from my_ai.models.attention import MultiHeadAttention


class MiniTransformerLM(nn.Module):
    """ Финальный класс нашей языковой модели """

    def __init__(self, vocab_size, block_size, n_embd=32, num_heads=4):
        super().__init__()
        self.block_size = block_size

        # 1. Таблица эмбеддингов символов (букв)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # 2. Позиционные эмбеддинги (чтобы ИИ знал, где какая буква стоит по счету)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 3. Наш блок многоголового внимания (4 головы по 8 каналов каждая = 32 скрытых признака)
        self.sa_heads = MultiHeadAttention(num_heads, head_size=n_embd // num_heads, n_embd=n_embd, block_size=block_size)

        # 4. Выходной слой для генерации предсказаний букв
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Переводим ID букв в смысловые векторы
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        # Создаем векторы позиций от 0 до T-1 и превращаем их в позиционные эмбеддинги
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)

        # Складываем смысл букв с их позициями в предложении!
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Пропускаем данные через механизм внимания (ИИ анализирует контекст)
        x = self.sa_heads(x)  # (B, T, n_embd)

        # Считаем финальные оценки (логиты) для следующей буквы
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Меняем форму матриц для расчета Cross-Entropy ошибки в PyTorch
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
