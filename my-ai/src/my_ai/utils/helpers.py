import torch


def get_device() -> torch.device:
    """
    Автоматически определяет лучшее доступное устройство для вычислений (GPU/CPU).
    Аналог фабричного метода в Java.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Поддержка ускорения на Mac M1/M2/M3/M4
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Считает количество обучаемых параметров (весов) в нейросети.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_config(config: dict, model_params_count: int, device: torch.device):
    """
    Красиво выводит конфигурацию проекта в консоль перед обучением.
    """
    print("=" * 50)
    print(" CONFIGURATION & ENVIRONMENT ")
    print("=" * 50)
    print(f"Device:           {device.type.upper()}")
    print(f"Model parameters: {model_params_count:,}")
    for key, value in config.items():
        print(f"{key:<17} {value}")
    print("=" * 50 + "\n")
