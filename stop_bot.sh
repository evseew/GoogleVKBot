#!/bin/bash

# Остановка бота
if [ -f bot.pid ]; then
    echo "Останавливаем бота по PID из файла..."
    PID=$(cat bot.pid)
    kill $PID
    rm bot.pid
    echo "Бот остановлен (PID: $PID)"
else
    echo "Файл bot.pid не найден, ищем процесс бота..."
    BOT_PID=$(ps aux | grep "python bot.py" | grep -v grep | awk '{print $2}')
    
    if [ -n "$BOT_PID" ]; then
        echo "Найден процесс бота с PID: $BOT_PID"
        kill $BOT_PID
        echo "Бот остановлен (PID: $BOT_PID)"
    else
        echo "Процесс бота не найден. Возможно, бот не запущен или запущен с другим именем файла."
    fi
fi 