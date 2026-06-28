import torch

from my_ai.dataset import CharacterTokenizer
from my_ai.models.transformer import MiniTransformerLM


def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    # 1. Используем тот же базовый текст, чтобы токенизатор воссоздал точно такой же словарь
    text = "мама мыла раму, а папа мыл машину. искусственный интеллект — это просто математика!"
    tokenizer = CharacterTokenizer(text)

    block_size = 8  # Контекст должен быть строго как при обучении

    # 2. Создаем «пустую» модель
    model = MiniTransformerLM(vocab_size=tokenizer.vocab_size, block_size=block_size)

    # 3. Загружаем сохраненные веса из файла model_weights.pth
    # weights_only=True — требование безопасности современных версий PyTorch
    model.load_state_dict(torch.load('model_weights.pth', map_location='cpu', weights_only=True))

    # 4. Переводим модель в режим оценки (eval), отключая учебные механизмы
    model.eval()

    # Превращаем ваш стартовый текст в массив чисел и добавляем размерность батча
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    print(f"Ваш промпт: '{prompt}'")
    print("ИИ генерирует продолжение...")

    generated_ids = []

    # Цикл генерации: предсказываем по одному символу за раз
    with torch.no_grad():  # Отключаем расчет градиентов (для инференса они не нужны и только тратят ОЗУ)
        for _ in range(max_new_tokens):
            # Если накопленный текст длиннее block_size (8 букв), берем только последние 8 символов
            context_cond = context[:, -block_size:]

            # Прогоняем контекст через нейросеть
            logits, _ = model(context_cond)

            # Нас интересуют оценки (logits) только для самого последнего сгенерированного символа
            logits = logits[:, -1, :]

            # Превращаем оценки в распределение вероятностей (от 0 до 1)
            probs = torch.softmax(logits, dim=-1)

            # Случайно выбираем следующий ID символа на основе их вероятностей (сэмплирование)
            next_id = torch.multinomial(probs, num_samples=1)

            # Приклеиваем новую букву к общему контексту, чтобы на следующем шаге ИИ видел и её тоже
            context = torch.cat((context, next_id), dim=1)
            generated_ids.append(next_id.item())

    # Декодируем список полученных ID обратно в буквы и склеиваем с промптом
    return prompt + tokenizer.decode(generated_ids)


if __name__ == '__main__':
    # Запускаем инференс. ВАЖНО: используйте буквы и слова, которые были в обучающем тексте!
    result = generate_text(prompt="мама ", max_new_tokens=80)
    print("\n--- Результат работы вашего ИИ ---")
    print(result)
