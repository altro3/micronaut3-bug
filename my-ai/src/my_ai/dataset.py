import torch


class CharacterTokenizer:
    def __init__(self, text: str):
        self.text = text
        # Собираем уникальные символы и сортируем их (наш алфавит)
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Словари для быстрой конвертации (аналог HashMap в Java)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, string: str) -> list[int]:
        # Строка -> список чисел
        return [self.stoi[c] for c in string]

    def decode(self, ids: list[int]) -> str:
        # Список чисел -> Строка
        return ''.join([itos_id for i in ids if (itos_id := self.itos.get(i)) is not None])

    def text_to_tensor(self) -> torch.Tensor:
        # Весь текст превращаем в один большой вектор чисел для PyTorch
        return torch.tensor(self.encode(self.text), dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    # block_size — длина контекста, batch_size — сколько примеров учим за раз
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Скользящие окна: X — входной текст, Y — тот же текст, сдвинутый на 1 символ вперед
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    return x, y
