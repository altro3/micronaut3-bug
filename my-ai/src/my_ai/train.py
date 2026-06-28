import math
import pickle
import random

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

        self.m_w = [create_zero_matrix(len(w), len(w[0])) for w, _ in model.get_parameters()]
        self.v_w = [create_zero_matrix(len(w), len(w[0])) for w, _ in model.get_parameters()]

        self.m_b = [[0.0] * len(b) for b, _ in model.get_biases()]
        self.v_b = [[0.0] * len(b) for b, _ in model.get_biases()]

    def step(self):
        self.t += 1
        for idx, (w, grad_w) in enumerate(self.model.get_parameters()):
            for i in range(len(w)):
                for j in range(len(w[0])):
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


def stable_cross_entropy_loss(logits: list[list[float]], targets: list[int]) -> tuple[float, list[list[float]]]:
    T = len(logits)
    vocab_size = len(logits[0])
    loss = 0.0
    grad_logits = create_zero_matrix(T, vocab_size)

    for t in range(T):
        row = logits[t]
        target_id = targets[t]

        max_val = max(row)
        sum_exps = sum(math.exp(x - max_val) for x in row)
        log_sum_exp = max_val + math.log(sum_exps)

        loss += log_sum_exp - row[target_id]

        probs = [math.exp(x - max_val) / sum_exps for x in row]
        for j in range(vocab_size):
            grad_logits[t][j] = probs[j]

        grad_logits[t][target_id] -= 1.0

    return loss / T, grad_logits


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
    for _ in range(300):
        random.shuffle(facts)
        article = " ".join(facts)

        qa = random.choice(qa_templates)

        prompt = f"Контекст: {article} Вопрос: {qa['q']} Ответ: {qa['a']}. "
        dataset_text += prompt

    tokenizer = CharacterTokenizer(dataset_text)
    data = tokenizer.encode(dataset_text)

    block_size = 36
    batch_size = 2
    max_iters = 5000

    model = PureTransformerLM(vocab_size=tokenizer.vocab_size, block_size=block_size, n_embd=24)
    optimizer = PureAdam(model, lr=0.002)

    print("=" * 50)
    print(" ОБУЧЕНИЕ ИСТИННОГО КОНТЕКСТНОГО ТРАНСФОРМЕРА ")
    print("=" * 50)

    loss = 0.0
    for step in range(max_iters):
        x_batch, y_batch = get_pure_batch(data, block_size, batch_size)
        total_step_loss = 0.0
        model.zero_grad()

        for b in range(batch_size):
            logits = model.forward(x_batch[b])
            current_loss, grad_logits = stable_cross_entropy_loss(logits, y_batch[b])
            total_step_loss += current_loss

            for t in range(len(grad_logits)):
                for j in range(len(grad_logits[t])):
                    grad_logits[t][j] /= batch_size
            model.backward(grad_logits)

        optimizer.step()
        if step % 200 == 0:
            print(f"Шаг {step:4d} | Ошибка (Loss): {total_step_loss / batch_size:.4f}")

    print("-" * 50)
    print("Обучение завершено! Модель поняла логику работы с контекстом.")
    with open('pure_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)


if __name__ == '__main__':
    main()
