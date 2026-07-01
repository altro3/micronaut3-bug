import os
from datasets import load_dataset

def main():
    print("[Python] Начинаем загрузку датасета русской Википедии с Hugging Face...")
    # Загружаем датасет, который ты нашёл
    dataset = load_dataset("Mikimi/ru-wikipedia-top-200k-full-text")

    # Создаем папку data, если её нет
    os.makedirs("data", exist_ok=True)
    output_path = "data/input.txt"

    print("[Python] Датасет загружен. Извлекаем тексты статей...")
    # Извлекаем колонку 'full_text' из сплита 'train'
    articles = dataset["train"]["full_text"]

    print(f"[Python] Всего нашли {len(articles)} статей. Записываем в {output_path}...")

    # Записываем всё в один огромный TXT-файл
    with open(output_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(articles):
            if text: # Проверяем, что текст не пустой
                f.write(text)
                f.write("\n\n") # Разделитель между статьями

            if (i + 1) % 2000 == 0:
                print(f"|-> Записано {i + 1} / {len(articles)} статей...")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Python] Готово! Итоговый файл input.txt собран. Размер: {file_size_mb:.2f} МБ")

if __name__ == "__main__":
    main()
