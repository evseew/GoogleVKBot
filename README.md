# Google Business Bot

Телеграм-бот для бизнес-аккаунта с векторной базой данных документов из Google Drive.

## Установка на VPS-сервер

### Быстрая установка через Git

1. Клонируйте репозиторий на сервер:
   ```
   git clone https://github.com/evseew/GoogleBusinessBot.git
   cd GoogleBusinessBot
   ```

2. Запустите скрипт установки:
   ```
   chmod +x server_setup.sh
   ./server_setup.sh
   ```

3. Настройте файл .env:
   ```
   nano .env
   ```
   Добавьте следующие строки:
   ```
   TELEGRAM_BOT_TOKEN=ваш_токен_бота
   OPENAI_API_KEY=ваш_ключ_openai
   ASSISTANT_ID=ваш_id_помощника
   GOOGLE_DRIVE_FOLDER_ID=id_папки_гугл_драйв
   ```

4. Добавьте файл `service-account-key.json` в корневую директорию

5. Запустите бота как systemd-сервис:
   ```
   sudo systemctl start google-business-bot
   sudo systemctl status google-business-bot
   ```

### Установка вручную

1. Создайте директорию и скопируйте файлы:
   ```
   mkdir -p ~/GoogleBusinessBot/logs/context_logs
   cd ~/GoogleBusinessBot
   # Скопируйте все файлы из репозитория
   ```

2. Создайте виртуальное окружение:
   ```
   python3 -m venv new_venv
   source new_venv/bin/activate
   pip install -r requirements.txt
   ```

3. Настройте файл .env и добавьте service-account-key.json

4. Настройте сервис и cron:
   ```
   sudo cp google-business-bot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable google-business-bot
   (crontab -l 2>/dev/null; cat crontab_entry.txt) | crontab -
   ```

## Управление ботом

- **Запуск**: `sudo systemctl start google-business-bot` или `./start_bot.sh`
- **Остановка**: `sudo systemctl stop google-business-bot` или `./stop_bot.sh`
- **Статус**: `sudo systemctl status google-business-bot` или `./control.sh status`
- **Перезапуск**: `sudo systemctl restart google-business-bot` или `./restart.sh`
- **Обновление базы**: `./update_db.sh`
- **Логи службы**: `journalctl -u google-business-bot -f`
- **Логи бота**: `tail -f logs/bot.log`

## Проверка работоспособности

1. Отправьте команду `/start` боту в Telegram
2. Проверьте логи на наличие ошибок: `tail -f logs/bot.log`
3. Проверьте статус службы: `sudo systemctl status google-business-bot`

## Устранение проблем

- **Бот не запускается**:
  - Проверьте логи: `journalctl -u google-business-bot -e`
  - Убедитесь, что .env содержит правильные ключи API
  - Проверьте права доступа: `chmod +x *.sh`

- **Проблемы с базой данных**:
  - Обновите базу вручную: `./update_db.sh`
  - Проверьте логи обновления: `cat logs/db_update.log`

- **Обновление из Git**:
  ```
  git pull
  sudo systemctl restart google-business-bot
  ```

## Команды бота

- `/start` - начать диалог и обновить базу знаний
- `/clear` - очистить историю диалога
- `/update` - обновить базу знаний вручную
- `/check_db` - проверить наличие базы знаний

## Мониторинг

Логи бота находятся в папке `logs`:
- Основной лог: `logs/bot.log`
- Лог обновления базы: `logs/db_update.log`

## Требования

- Python 3.7+
- aiogram 3.x
- openai
- python-dotenv

## Настройка интеграции с Telegram Business

Для подключения бота к бизнес-аккаунту Telegram:

1. **Создайте бизнес-аккаунт в Telegram**:
   - Откройте настройки Telegram
   - Выберите "Telegram Premium" > "Telegram Business"
   - Следуйте инструкциям для создания бизнес-профиля

2. **Подключите бота вручную в настройках бизнес-аккаунта**:
   - В настройках бизнес-аккаунта выберите "Чат-бот"
   - Нажмите "Подключить бота" и выберите вашего бота

3. **Настройте дополнительные бизнес-функции в интерфейсе Telegram**:
-  - После подключения бота можно настроить бизнес-функции
-  - Отправьте боту команду `/business_hours` для настройки часов работы
-  - Отправьте боту команду `/business_greeting Текст` для настройки приветственного сообщения
+  - Все бизнес-настройки (часы работы, приветствия, ответы) выполняются через 
+    меню Telegram Business в настройках вашего аккаунта
+  - Бот автоматически поддерживает бизнес-режим без дополнительной настройки

- ## Бизнес-команды бота
- 
- - `/business_hours` - установить стандартные часы работы (Пн-Пт, 9:00-18:00)
- - `/business_greeting Текст` - установить приветственное сообщение для клиентов