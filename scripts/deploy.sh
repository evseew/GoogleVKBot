#!/bin/bash

# Конфигурация
SERVER="root@185.125.203.247"
REMOTE_DIR="/root/TestBot"

# Создаем необходимые директории на сервере
ssh $SERVER "mkdir -p $REMOTE_DIR/logs"

# Копируем файлы
scp bot.py $SERVER:$REMOTE_DIR/
scp google_drive_sync.py $SERVER:$REMOTE_DIR/
scp requirements.txt $SERVER:$REMOTE_DIR/
scp .env $SERVER:$REMOTE_DIR/
scp systemd/*.service $SERVER:/etc/systemd/system/

# Устанавливаем зависимости на сервере
ssh $SERVER "cd $REMOTE_DIR && source venv/bin/activate && pip install -r requirements.txt"

# Перезапускаем сервисы
ssh $SERVER "systemctl daemon-reload && systemctl restart google-drive-sync && systemctl restart telegram-bot"

# Проверяем статус
ssh $SERVER "systemctl status google-drive-sync && systemctl status telegram-bot" 