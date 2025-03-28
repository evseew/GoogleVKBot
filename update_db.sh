#!/bin/bash
# Обновление векторной базы данных
echo "Обновление базы данных..."

# Активируем виртуальное окружение
echo "Активируем окружение: $(pwd)/new_venv"
source "$(pwd)/new_venv/bin/activate"

source .env

# Создаем директорию для логов, если она не существует
mkdir -p logs

python rebuild_db_fixed.py > logs/db_update.log 2>&1
echo "Обновление завершено, смотрите logs/db_update.log" 

# Если есть строка типа
# python -c "from langchain_openai import OpenAIEmbeddings; embeddings = OpenAIEmbeddings(model='text-embedding-3-small')"
# замените ее на
python -c "from langchain_openai import OpenAIEmbeddings; embeddings = OpenAIEmbeddings(model='text-embedding-3-large')" 

import signal
import sys

def signal_handler(sig, frame):
    logging.info("Получен сигнал завершения работы")
    # Закрываем соединения и ресурсы
    # Удаляем PID файлы
    if os.path.exists('bot.pid'):
        os.remove('bot.pid')
    # Другие файлы PID
    for pid_file in glob.glob('bot *.pid'):
        os.remove(pid_file)
    logging.info("Бот корректно завершил работу")
    sys.exit(0)

# Регистрируем обработчик
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler) 