from dataclasses import dataclass

import torch

from my_ai.dataset import CharacterTokenizer, get_batch
from my_ai.models.transformer import MiniTransformerLM
from my_ai.utils.helpers import get_device, count_parameters, print_config


@dataclass
class TrainConfig:
    block_size: int = 8
    batch_size: int = 4
    max_iters: int = 3000
    learning_rate: float = 1e-3


def main():
    # Создаем объект конфигурации с жесткими типами данных
    config = TrainConfig()

    text = "мама мыла раму, а папа мыл машину. искусственный интеллект — это просто математика!"

    tokenizer = CharacterTokenizer(text)
    data = tokenizer.text_to_tensor()

    device = get_device()

    # Обратите внимание, теперь мы обращаемся к полям через точку, как в Java: config.block_size
    model = MiniTransformerLM(vocab_size=tokenizer.vocab_size, block_size=config.block_size)
    model = model.to(device)

    # Передаем объект в хелпер, превратив его в словарь только для вывода на экран (через __dict__)
    print_config(config.__dict__, count_parameters(model), device)

    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("Начинаем процесс обучения нашего ИИ...")

    # Теперь Идея точно знает, что config.max_iters — это строго int, и ворнинг исчезнет!
    for steps in range(config.max_iters):
        xb, yb = get_batch(data, config.block_size, config.batch_size)

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if steps % 500 == 0:
            print(f"Шаг {steps:4d}: текущая ошибка (loss) = {loss.item():.4f}")

    print("\nОбучение полностью завершено!")
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Веса модели успешно сохранены в файл 'model_weights.pth'!")


if __name__ == '__main__':
    main()
