#!/bin/bash

# Определяем директорию скрипта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Функция для запуска бота
start_bot() {
    if pgrep -f "python.*bot.py" > /dev/null; then
        echo "Бот уже запущен!"
    else
        echo "Запускаем бота..."
        "${SCRIPT_DIR}/start_bot.sh" &
        echo "Бот запущен!"
    fi
}

# Функция для остановки бота
stop_bot() {
    if pgrep -f "python.*bot.py" > /dev/null; then
        echo "Останавливаем бота..."
        pkill -f "python.*bot.py"
        echo "Бот остановлен!"
    else
        echo "Бот не запущен!"
    fi
}

# Функция для перезапуска бота
restart_bot() {
    stop_bot
    sleep 2
    start_bot
}

# Функция для проверки статуса бота
status_bot() {
    if pgrep -f "python.*bot.py" > /dev/null; then
        echo "Бот запущен!"
    else
        echo "Бот не запущен!"
    fi
}

# Обработка аргументов командной строки
case "$1" in
    start)
        start_bot
        ;;
    stop)
        stop_bot
        ;;
    restart)
        restart_bot
        ;;
    status)
        status_bot
        ;;
    *)
        echo "Использование: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

exit 0
