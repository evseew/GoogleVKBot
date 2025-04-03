#!/bin/bash
# Генерация файла requirements.txt с полным списком зависимостей
echo "Создание полного списка зависимостей..."
source new_venv/bin/activate
pip freeze > requirements.txt
echo "✅ Файл requirements.txt обновлен с полным списком зависимостей" 