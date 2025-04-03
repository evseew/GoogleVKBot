#!/bin/bash
# Скрипт настройки сервера для Google Business Bot

# Выводим информацию о начале установки
echo "=== Начало установки Google Business Bot ==="
echo "Дата: $(date)"

# 1. Создаем директории
echo "Создание директорий..."
mkdir -p logs
mkdir -p logs/context_logs

# 2. Устанавливаем Python 3.10 (для Ubuntu/Debian)
echo "Установка Python и необходимых пакетов..."
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip git

# 3. Создаем виртуальное окружение
echo "Создание виртуального окружения..."
python3.10 -m venv new_venv
source new_venv/bin/activate

# 4. Устанавливаем зависимости
echo "Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Делаем скрипты исполняемыми
echo "Настройка прав доступа для скриптов..."
chmod +x start_bot.sh stop_bot.sh update_db.sh control.sh restart.sh

# 6. Создаем сервис systemd
echo "Настройка systemd сервиса..."
sudo cp google-business-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable google-business-bot

# 7. Настройка crontab для обновления базы
echo "Настройка планировщика для обновления базы данных..."
(crontab -l 2>/dev/null; cat crontab_entry.txt) | crontab -

# 8. Проверка наличия необходимых файлов
echo "Проверка необходимых файлов..."
if [ ! -f .env ]; then
    echo "ВНИМАНИЕ: Файл .env не найден! Необходимо создать его и добавить все переменные окружения."
fi

if [ ! -f service-account-key.json ]; then
    echo "ВНИМАНИЕ: Файл service-account-key.json не найден! Необходимо добавить его для работы с Google Drive."
fi

# 9. Создание начальной базы данных
echo "Создание начальной базы данных..."
./update_db.sh

echo "=== Установка завершена ==="
echo "Чтобы запустить бота, выполните: sudo systemctl start google-business-bot"
echo "Для проверки статуса: sudo systemctl status google-business-bot"
echo "Для просмотра логов: journalctl -u google-business-bot -f" 