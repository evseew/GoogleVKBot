#!/bin/bash

# Перейти в директорию бота
cd /root/GoogleBusinessTelegramBot || exit 1

# Активировать виртуальное окружение
source new_venv/bin/activate || exit 1 

# Получить последние изменения с удаленного репозитория, не применяя их
git fetch

# Проверить, есть ли изменения на удаленном сервере (сравнить локальный HEAD с удаленным)
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse @{u}) # @{u} означает upstream (обычно origin/main или origin/master)

if [ "$LOCAL" != "$REMOTE" ]; then
  echo "$(date): Найдены обновления в Git. Запускаю обновление..." >> logs/auto_update.log
  
  # Загрузить изменения
  git pull >> logs/auto_update.log 2>&1
  if [ $? -ne 0 ]; then
      echo "$(date): ОШИБКА: Не удалось загрузить изменения (git pull)" >> logs/auto_update.log
      exit 1 # Выйти, если pull не удался
  fi
  
  echo "$(date): Обновления успешно загружены." >> logs/auto_update.log

  # (Опционально, но рекомендуется) Установить/обновить зависимости, если requirements.txt изменился
  # Проверяем, изменился ли requirements.txt в последнем коммите (или с момента последнего pull)
  if git diff HEAD@{1}..HEAD --quiet -- requirements.txt; then
      echo "$(date): requirements.txt не изменился." >> logs/auto_update.log
  else
      echo "$(date): requirements.txt изменился. Устанавливаю зависимости..." >> logs/auto_update.log
      pip install -r requirements.txt >> logs/auto_update.log 2>&1
      if [ $? -ne 0 ]; then
          echo "$(date): ОШИБКА: Не удалось установить зависимости (pip install)" >> logs/auto_update.log
          # Решите, стоит ли прерывать перезапуск, если зависимости не установились
          # exit 1 
      fi
  fi

  # Перезапустить бота
  echo "$(date): Перезапускаю бота..." >> logs/auto_update.log
  ./restart.sh >> logs/auto_update.log 2>&1 

  echo "$(date): Процесс автообновления завершен." >> logs/auto_update.log
else
  # Если обновлений нет, ничего не делать
  : 
fi

exit 0 