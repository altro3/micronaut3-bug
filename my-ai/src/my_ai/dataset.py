import random


class CharacterTokenizer:
    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, string: str) -> list[int]:
        return [self.stoi[c] for c in string]

    def decode(self, ids: list[int]) -> str:
        return ''.join([self.itos[i] for i in ids])


def get_pure_batch(data: list[int], block_size: int, batch_size: int):
    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        start_idx = random.randint(0, len(data) - block_size - 1)
        x_batch.append(data[start_idx: start_idx + block_size])
        y_batch.append(data[start_idx + 1: start_idx + block_size + 1])

    return x_batch, y_batch
