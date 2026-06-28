import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ Одна голова механизма внимания (Self-Attention Head) """

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        # Создаем три линейных слоя (матрицы весов) для Q, K и V
        # В Java это были бы три объекта класса LinearLayer(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Маска триггера (нижнетреугольная матрица), чтобы модель не подглядывала в будущее
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Входной тензор x имеет размер (B, T, C)
        # B — батч, T — длина контекста (время), C — размерность эмбеддинга
        B, T, C = x.shape

        # Вычисляем Ключи и Запросы для всех символов в контексте
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Считаем веса внимания: перемножаем Запросы и Ключи.
        # Матричное умножение: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Делим на корень из размерности (head_size ** -0.5) для стабильности математики
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # Важнейший этап: каузальная маска. Запрещаем модели смотреть на будущие буквы.
        # Заменяем все нули в верхней части матрицы на минус бесконечность (-inf)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Пропускаем через Softmax: минус бесконечности превратятся в 0% вероятности,
        # а реальные связи букв — в красивые веса от 0 до 1
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Вычисляем Значения (смысловую нагрузку букв)
        v = self.value(x)  # (B, T, head_size)

        # Финальный аккорд: умножаем веса внимания на Значения
        # Каждое слово забирает в себя контекст прошлых слов, с которыми совпало
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Несколько голов Self-Attention, работающих параллельно """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        # Создаем список из независимых голов внимания
        # nn.ModuleList — это аналог ArrayList<nn.Module> в Java, о котором должен знать PyTorch
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        # Линейный слой для объединения результатов всех голов обратно в один вектор
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        # Запускаем каждую голову на входных данных x и объединяем их выходы по последней оси (dim=-1)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Пропускаем через финальную проекцию
        return self.proj(out)
