#!/bin/bash

# Создаем виртуальное окружение если его нет
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Активируем виртуальное окружение
source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Создаем необходимые директории
mkdir -p logs
mkdir -p data
mkdir -p vector_store

# Проверяем наличие .env файла
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "TELEGRAM_BOT_TOKEN=" > .env
    echo "OPENAI_API_KEY=" >> .env
    echo "ASSISTANT_ID=" >> .env
    echo "Please fill in the .env file with your credentials"
fi

echo "Local environment setup complete!" 