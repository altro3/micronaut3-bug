import sys

from setuptools import Extension, setup

# Настройки для Си-расширения
module = Extension(
    'my_ai.utils.matrix_math_c',  # Имя будущей либы
    sources=['utils/matrix_math.c']  # Где лежит наш Си-код
)

# Перенаправляем аргументы, чтобы запустить сборку прямо из кода
sys.argv = ['compile.py', 'build_ext', '--inplace']

print("Начинаем автоматическую сборку Си-библиотеки средствами Windows...")
setup(
    name='MatrixMathC',
    version='1.0',
    ext_modules=[module]
)
print("Сборка успешно завершена!")
