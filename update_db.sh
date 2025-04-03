#!/bin/bash
# Обновление векторной базы данных
echo "Обновление базы данных..."

# Активируем виртуальное окружение
echo "Активируем окружение: $(pwd)/new_venv"
source "$(pwd)/new_venv/bin/activate"

# Загружаем переменные окружения
source .env

# Создаем директорию для логов, если она не существует
mkdir -p logs

# Запускаем скрипт обновления базы
python rebuild_db_fixed.py > logs/db_update.log 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Обновление завершено успешно, смотрите logs/db_update.log"
else
    echo "Ошибка при обновлении базы, проверьте logs/db_update.log"
fi

exit $EXIT_CODE 