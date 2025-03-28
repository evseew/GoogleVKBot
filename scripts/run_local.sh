#!/bin/bash

# Активируем виртуальное окружение
source venv/bin/activate

# Запускаем оба скрипта в разных терминалах
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && source venv/bin/activate && python3 google_drive_sync.py"'
    osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && source venv/bin/activate && python3 bot.py"'
else
    # Linux
    gnome-terminal -- bash -c "source venv/bin/activate && python3 google_drive_sync.py; exec bash"
    gnome-terminal -- bash -c "source venv/bin/activate && python3 bot.py; exec bash"
fi 