import pickle
import random

from my_ai.utils.matrix_math import softmax_row


def ask_ai_direct(fact: str, question: str, max_new_tokens: int = 15) -> str:
    block_size = 16
    temperature = 0.1

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('pure_model.pkl', 'rb') as f:
        model = pickle.load(f)

    prompt = f"Контекст: {fact} Вопрос: {question} Ответ: "

    context = tokenizer.encode(prompt)[-block_size:]
    generated_ids = []

    for _ in range(max_new_tokens):
        context_cond = context[-block_size:]
        logits = model.forward(context_cond)
        last_letter_logits = list(logits[-1])

        if temperature != 1.0:
            last_letter_logits = [x / temperature for x in last_letter_logits]

        probs = softmax_row(last_letter_logits)

        r = random.random()
        cumulative_prob = 0.0
        next_id = 0
        for idx, p in enumerate(probs):
            cumulative_prob += p
            if r <= cumulative_prob:
                next_id = idx
                break

        predicted_token = tokenizer.itos[next_id]

        if "[EOS]" in predicted_token or "Контекст" in predicted_token or "Модель" in predicted_token:
            break

        context.append(next_id)
        generated_ids.append(next_id)

    # Декодируем весь сгенерированный текст целиком
    full_generation = tokenizer.decode(generated_ids).strip()

    if "[EOS]" in full_generation:
        full_generation = full_generation.split("[EOS]")[0]

    for stop_word in ["Контекст:", "Модель", "Вопрос:", "Оптимизато"]:
        if stop_word in full_generation:
            full_generation = full_generation.split(stop_word)[0]

    return full_generation.strip()


if __name__ == '__main__':
    print("=" * 50)
    print(" ФИНАЛЬНЫЙ ТЕСТ IN-CONTEXT LEARNING ")
    print("=" * 50)

    tests = [
        {
            "fact": "Искусственный интеллект работает на чистой математике.",
            "q": "На чем работает искусственный интеллект?"
        },
        {
            "fact": "Трансформер признан лучшей архитектурой для работы с текстом.",
            "q": "Какая архитектура признана лучшей?"
        }
    ]

    for test in tests:
        answer = ask_ai_direct(test["fact"], test["q"])
        print(f"Контекст: {test['fact']}")
        print(f"Вопрос:   {test['q']}")
        print(f"Ответ ИИ: {answer}")
        print("-" * 50)
