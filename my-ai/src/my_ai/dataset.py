import random


class CharacterTokenizer:
    def __init__(self, text: str, num_merges: int = 150):
        self.num_merges = num_merges

        self.vocab = sorted(list(set(text)))

        if "[UNK]" not in self.vocab: self.vocab.append("[UNK]")
        if "[EOS]" not in self.vocab: self.vocab.append("[EOS]")

        current_tokens = list(text)
        self.merges = {}

        for _ in range(num_merges):
            pairs = {}
            for i in range(len(current_tokens) - 1):
                p = (current_tokens[i], current_tokens[i + 1])
                if "[EOS]" in p or "[UNK]" in p:
                    continue
                pairs[p] = pairs.get(p, 0) + 1

            if not pairs:
                break

            best_pair = max(pairs, key=lambda k: pairs[k])
            if pairs[best_pair] < 2:
                break

            new_token = best_pair[0] + best_pair[1]

            self.merges[best_pair] = new_token
            self.vocab.append(new_token)

            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if i < len(current_tokens) - 1 and (current_tokens[i], current_tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
            current_tokens = new_tokens

        self.vocab_size = len(self.vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}

    def encode(self, string: str) -> list[int]:
        tokens = []
        for char in string:
            if char in self.stoi:
                tokens.append(char)
            else:
                tokens.append("[UNK]")

        for pair, new_token in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.stoi[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.itos[i] for i in ids])


def get_pure_batch(data: list[int], block_size: int, batch_size: int):
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        start_idx = random.randint(0, len(data) - block_size - 1)
        x_batch.append(data[start_idx: start_idx + block_size])
        y_batch.append(data[start_idx + 1: start_idx + block_size + 1])
    return x_batch, y_batch
