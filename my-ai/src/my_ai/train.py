import math
import pickle
import random
import time
from datetime import datetime

from my_ai.dataset import CharacterTokenizer, get_pure_batch
from my_ai.models.transformer import PureTransformerLM
from my_ai.utils.matrix_math import create_zero_matrix


class PureAdam:
    def __init__(self, model: PureTransformerLM, lr=0.001):
        self.model = model
        self.lr = lr
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        self.m_w = [create_zero_matrix(len(w), len(w[0]) if len(w) > 0 else 0) for w, _ in model.get_parameters()]
        self.v_w = [create_zero_matrix(len(w), len(w[0]) if len(w) > 0 else 0) for w, _ in model.get_parameters()]

        self.m_b = [[0.0] * len(b) for b, _ in model.get_biases()]
        self.v_b = [[0.0] * len(b) for b, _ in model.get_biases()]

    def step(self):
        self.t += 1
        for idx, (w, grad_w) in enumerate(self.model.get_parameters()):
            rows = len(w)
            cols = len(w[0]) if rows > 0 else 0

            for i in range(rows):
                for j in range(cols):
                    g = grad_w[i][j]
                    self.m_w[idx][i][j] = self.beta1 * self.m_w[idx][i][j] + (1 - self.beta1) * g
                    self.v_w[idx][i][j] = self.beta2 * self.v_w[idx][i][j] + (1 - self.beta2) * (g ** 2)

                    m_corrected = self.m_w[idx][i][j] / (1 - self.beta1 ** self.t)
                    v_corrected = self.v_w[idx][i][j] / (1 - self.beta2 ** self.t)
                    w[i][j] -= self.lr * m_corrected / (math.sqrt(v_corrected) + self.eps)

        for idx, (b, grad_b) in enumerate(self.model.get_biases()):
            for j in range(len(b)):
                g = grad_b[j]
                self.m_b[idx][j] = self.beta1 * self.m_b[idx][j] + (1 - self.beta1) * g
                self.v_b[idx][j] = self.beta2 * self.v_b[idx][j] + (1 - self.beta2) * (g ** 2)

                m_corrected = self.m_b[idx][j] / (1 - self.beta1 ** self.t)
                v_corrected = self.v_b[idx][j] / (1 - self.beta2 ** self.t)
                b[j] -= self.lr * m_corrected / (math.sqrt(v_corrected) + self.eps)


def stable_cross_entropy_loss(logits: list[list[float]], targets: list[int], vocab_size: int) -> tuple[float, list[list[float]]]:
    T = len(logits)
    loss = 0.0
    grad_logits = [[0.0 for _ in range(vocab_size)] for _ in range(T)]

    for t in range(T):
        row = logits[t]
        target_id = targets[t]

        if target_id >= vocab_size:
            target_id = vocab_size - 1

        max_val = max(row)
        sum_exps = sum(math.exp(x - max_val) for x in row)
        log_sum_exp = max_val + math.log(sum_exps)

        loss += log_sum_exp - row[target_id]

        probs = [math.exp(x - max_val) / sum_exps for x in row]
        for j in range(vocab_size):
            grad_logits[t][j] = probs[j]

        grad_logits[t][target_id] -= 1.0

    return loss / T, grad_logits


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}м {seconds:.1f}с"


def main():
    facts = [
        "Искусственный интеллект работает на чистой математике.",
        "Основой любого современного ИИ является нейросеть.",
        "Нейросеть состоит из математических слоев и весов.",
        "Модель ИИ обучается через обратное распространение ошибки.",
        "Ошибку вычислений считает функция кросс-энтропии.",
        "Оптимизатор Адам обновляет веса на основе градиентов.",
        "Трансформер признан лучшей архитектурой для работы с текстом."
    ]

    qa_templates = [
        {"q": "На чем работает искусственный интеллект?", "a": "на чистой математике"},
        {"q": "Что является основой ИИ?", "a": "нейросеть"},
        {"q": "Из чего состоит нейросеть?", "a": "из математических слоев и весов"},
        {"q": "Через что обучается модель?", "a": "через обратное распространение ошибки"},
        {"q": "Что считает ошибку вычислений?", "a": "функция кросс-энтропии"},
        {"q": "Какой оптимизатор обновляет веса?", "a": "Адам"},
        {"q": "Какая архитектура признана лучшей?", "a": "Трансформер"}
    ]

    dataset_text = ""
    for _ in range(400):
        random.shuffle(facts)
        article = " ".join(facts)
        qa = random.choice(qa_templates)
        prompt = f"Контекст: {article} Вопрос: {qa['q']} Ответ: {qa['a']}[EOS]"
        dataset_text += prompt

    tokenizer = CharacterTokenizer(dataset_text, num_merges=150)
    data = tokenizer.encode(dataset_text)

    block_size = 16
    batch_size = 2
    max_iters = 6000

    model = PureTransformerLM(vocab_size=tokenizer.vocab_size, block_size=block_size, n_embd=24)
    optimizer = PureAdam(model, lr=0.002)

    # Получаем и форматируем точное время старта процесса
    now_start = datetime.now()
    start_time_str = now_start.strftime("%H:%M:%S")

    print("=" * 70)
    print(" ЗАПУСК СКОРОСТНОГО BPE ТРАНСФОРМЕРА С ТАЙМЕРАМИ ")
    print(f" Размер словаря: {tokenizer.vocab_size} токенов | Контекст: {block_size} | Батч: {batch_size}")
    print(f" Время старта обучения (системное): {start_time_str}")
    print("=" * 70)

    start_train_time = time.time()
    last_log_time = time.time()
    log_interval = 200

    for step in range(max_iters + 1):
        x_batch, y_batch = get_pure_batch(data, block_size, batch_size)
        total_step_loss = 0.0
        model.zero_grad()

        for b in range(batch_size):
            logits = model.forward(x_batch[b])
            current_loss, grad_logits = stable_cross_entropy_loss(logits, y_batch[b], tokenizer.vocab_size)
            total_step_loss += current_loss

            for t in range(len(grad_logits)):
                for j in range(len(grad_logits[t])):
                    grad_logits[t][j] /= batch_size
            model.backward(grad_logits)

        optimizer.step()

        if step % log_interval == 0 and step > 0:
            current_time = time.time()
            elapsed_interval = current_time - last_log_time
            speed = log_interval / elapsed_interval
            seconds_per_it = elapsed_interval / log_interval
            total_elapsed = current_time - start_train_time

            remaining_steps = max_iters - step
            estimated_remaining_time = remaining_steps * seconds_per_it

            print(
                f"Шаг {step:4d}/{max_iters} | "
                f"Loss: {total_step_loss / batch_size:.4f} | "
                f"Скорость: {speed:.2f} it/s ({seconds_per_it * 1000:.1f} ms/it) | "
                f"Прошло: {format_time(total_elapsed)} | "
                f"Осталось: {format_time(estimated_remaining_time)}"
            )

            last_log_time = current_time

    # Конец обучения — получаем точное время завершения
    end_train_time = time.time()
    now_end = datetime.now()
    end_time_str = now_end.strftime("%H:%M:%S")

    total_time = end_train_time - start_train_time

    print("=" * 70)
    print("Обучение полностью завершено!")
    print(f" Время начала:   {start_time_str}")
    print(f" Время окончания: {end_time_str}")
    print(f" Общее чистое время работы алгоритма: {format_time(total_time)}")
    print("=" * 70)

    with open('pure_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)


if __name__ == '__main__':
    main()
