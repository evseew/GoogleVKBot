#!/bin/bash
# Скрипт настройки сервера

# 1. Создаем директории
mkdir -p ~/GoogleBusinessBot/logs

# 2. Копируем файлы (если вы используете git)
# cd ~/GoogleBusinessBot
# git clone YOUR_REPOSITORY .

# 3. Устанавливаем Python 3.10 (для Ubuntu/Debian)
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip

# 4. Создаем виртуальное окружение
python3.10 -m venv new_venv
source new_venv/bin/activate

# 5. Устанавливаем зависимости
pip install -r requirements.txt

# 6. Делаем скрипты исполняемыми
chmod +x start_bot.sh stop_bot.sh update_db.sh

# 7. Создаем сервис systemd
# sudo cp google-business-bot.service /etc/systemd/system/
# sudo systemctl daemon-reload
# sudo systemctl enable google-business-bot
# sudo systemctl start google-business-bot

echo "Настройка сервера завершена. Проверьте файл .env перед запуском бота!" 