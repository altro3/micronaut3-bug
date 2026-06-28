import pickle
import random

from my_ai.utils.matrix_math import softmax_row


def ask_ai_with_context(article: str, question: str, max_new_tokens: int = 30) -> str:
    block_size = 36
    temperature = 0.2

    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('pure_model.pkl', 'rb') as f:
        model = pickle.load(f)

    prompt = f"Контекст: {article} Вопрос: {question} Ответ: "

    context = tokenizer.encode(prompt)
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

        if tokenizer.itos[next_id] == '.':
            break

        context.append(next_id)
        generated_ids.append(next_id)

    return tokenizer.decode(generated_ids).strip()


if __name__ == '__main__':
    custom_article = (
        "Трансформер признан лучшей архитектурой для работы с текстом. "
        "Оптимизатор Адам обновляет веса на основе градиентов. "
        "Искусственный интеллект работает на чистой математике."
    )

    test_questions = [
        "На чем работает искусственный интеллект?",
        "Какая архитектура признана лучшей?"
    ]

    print("=" * 50)
    print(" ТЕСТИРОВАНИЕ НАСТОЯЩЕГО IN-CONTEXT LEARNING ")
    print("=" * 50)

    for q in test_questions:
        answer = ask_ai_with_context(custom_article, q)
        print(f"Статья в промпте: {custom_article}")
        print(f"Вопрос: {q}")
        print(f"Ответ ИИ: {answer}")
        print("-" * 50)
