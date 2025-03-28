#!/bin/bash

# Выводим полный путь к виртуальному окружению
echo "Активируем окружение: $(pwd)/new_venv"

# Активация окружения
source "$(pwd)/new_venv/bin/activate"

# Устанавливаем langchain-huggingface
pip install langchain-huggingface

# Проверяем, что библиотека установлена
# Проверка окружения
python3 -c "import openai; print('✅ OpenAI API готов к использованию')"

# Запуск бота в фоновом режиме с логированием
echo "Запуск бота..."
source .env
mkdir -p logs
nohup python3 bot.py > logs/bot.log 2>&1 &
echo $! > bot.pid
echo "Бот запущен, PID: $(cat bot.pid)"

tail -f logs/bot.log