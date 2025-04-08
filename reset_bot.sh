#!/bin/bash

# Определяем директорию скрипта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CONFIG_DIR="$SCRIPT_DIR/logs/context_logs"

# Вывод информации
echo "🔄 Начинаю полный сброс бота: $(date)"

# Останавливаем бота через systemd, если он запущен как сервис
if systemctl is-active --quiet google-business-bot; then
    echo "🛑 Останавливаю сервис бота..."
    sudo systemctl stop google-business-bot
else
    # Останавливаем бота напрямую, если он запущен как процесс
    if [ -f "$SCRIPT_DIR/bot.pid" ]; then
        echo "🛑 Останавливаю процесс бота..."
        "$SCRIPT_DIR/stop_bot.sh"
    fi
fi

# Очистка логов
echo "🧹 Удаляю файлы логов..."
mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"
rm -f "$LOG_DIR/bot.log"
rm -f "$CONFIG_DIR"/*

# Очистка контекстов
echo "🗑️ Удаляю сохраненные контексты OpenAI..."
# Здесь нам не нужно удалять файлы физически, так как они будут пересозданы при следующем запуске

# Запускаем бота через systemd
if systemctl list-unit-files | grep -q google-business-bot; then
    echo "🚀 Запускаю бота через systemd..."
    sudo systemctl start google-business-bot
    sleep 3
    STATUS=$(systemctl is-active google-business-bot)
    if [ "$STATUS" = "active" ]; then
        echo "✅ Бот успешно запущен!"
    else
        echo "❌ Ошибка запуска бота! Статус: $STATUS"
        echo "📋 Лог ошибки:"
        sudo journalctl -u google-business-bot -n 20
    fi
else
    echo "🚀 Запускаю бота напрямую..."
    "$SCRIPT_DIR/start_bot.sh"
fi

echo "✅ Бот перезапущен с чистой историей: $(date)"
echo "Теперь все диалоги начнутся с чистого листа!" 