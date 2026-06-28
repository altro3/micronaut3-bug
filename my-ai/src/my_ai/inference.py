import pickle
import random

from my_ai.dataset import CharacterTokenizer
from my_ai.models.transformer import PureTransformerLM
from my_ai.utils.matrix_math import softmax_row


def generate_text(prompt: str, max_new_tokens: int = 60, temperature: float = 0.8) -> str:
    text = "мама мыла раму, а папа мыл машину. искусственный интеллект — это просто математика!"
    tokenizer = CharacterTokenizer(text)

    block_size = 8

    with open('pure_model.pkl', 'rb') as f:
        model: PureTransformerLM = pickle.load(f)

    print(f"Ваш стартовый промпт: '{prompt}'")
    print("Самодельный ИИ генерирует текст...")

    context = tokenizer.encode(prompt)
    generated_ids = []

    for _ in range(max_new_tokens):
        context_cond = context[-block_size:]

        logits = model.forward(context_cond)
        last_letter_logits = logits[-1]

        if temperature != 1.0:
            last_letter_logits = [line / temperature for line in last_letter_logits]

        probs = softmax_row(last_letter_logits)

        r = random.random()
        cumulative_prob = 0.0
        next_id = 0
        for idx, p in enumerate(probs):
            cumulative_prob += p
            if r <= cumulative_prob:
                next_id = idx
                break

        context.append(next_id)
        generated_ids.append(next_id)

    return prompt + tokenizer.decode(generated_ids)


if __name__ == '__main__':
    result = generate_text(prompt="мама ", max_new_tokens=60, temperature=0.2)
    print("\n--- Финальный результат работы вашего ИИ (Без библиотек) ---")
    print(result)
