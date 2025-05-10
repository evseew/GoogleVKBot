import sys
import os
import time as time_module
import asyncio
import logging
import datetime
import glob
import io
from io import BytesIO
import signal
from collections import deque, defaultdict
from typing import Optional # Добавляем импорт Optional
from datetime import datetime, timedelta, time as dt_time
import pytz # Добавим эту строку в начало файла, где импорты
import shutil
import requests

# --- Dependency Imports ---
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEvent, VkBotEventType
# ВАЖНО: vk_api.VkBotLongPoll - СИНХРОННЫЙ. Для асинхронной работы нужен
# либо другой механизм (Callback API), либо асинхронная библиотека VK,
# либо запуск LongPoll в отдельном потоке. Текущая реализация с asyncio.create_task
# в цикле longpoll.listen() будет БЛОКИРОВАТЬСЯ.
# Пример асинхронной библиотеки (если доступна и подходит):
# from vkbottle import Bot, Message
# from vkbottle.bot import BotLabeler

import openai
import chromadb # Используем напрямую для большей гибкости
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import docx
import PyPDF2

# LangChain components for specific tasks
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter # Добавлен MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
# VK API Settings
VK_GROUP_TOKEN = os.getenv("VK_GROUP_TOKEN")
VK_GROUP_ID = os.getenv("VK_GROUP_ID")
VK_API_VERSION = os.getenv("VK_API_VERSION", "5.199")

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536 # Уточните, если используете другую размерность

# Google Drive Settings
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", 'service-account-key.json')
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# User/Manager IDs
# Убедитесь, что ID корректные целые числа
try:
    ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID"))
except (TypeError, ValueError):
    raise ValueError("❌ Ошибка: ADMIN_USER_ID должен быть числом в .env!")

try:
    raw_manager_ids = os.getenv("MANAGER_USER_IDS", "").split(',')
    MANAGER_USER_IDS = [int(id_str) for id_str in raw_manager_ids if id_str.strip()]
except ValueError:
     raise ValueError("❌ Ошибка: MANAGER_USER_IDS в .env должны быть числами, разделенными запятыми!")

# Vector Store Settings
VECTOR_DB_BASE_PATH = "./local_vector_db" # Базовая директория для всех версий БД
ACTIVE_DB_INFO_FILE = "active_db_info.txt" # Файл, хранящий имя активной поддиректории БД
VECTOR_DB_COLLECTION_NAME = "documents_collection" # Имя коллекции ChromaDB
RELEVANT_CONTEXT_COUNT = 3 # Количество релевантных чанков для контекста

# Bot Behavior Settings
MESSAGE_LIFETIME_DAYS = 100 # Срок хранения истории сообщений (в памяти)
MESSAGE_COOLDOWN_SECONDS = 3 # Минимальный интервал между сообщениями от пользователя
MESSAGE_BUFFER_SECONDS = 4 # Время ожидания для буферизации сообщений перед обработкой
MANAGER_ACTIVE_TIMEOUT_SECONDS = 86400 # 24 часа - время молчания бота после команды менеджера
LOG_RETENTION_SECONDS = 86400 # 24 часа - время хранения логов контекста
OPENAI_RUN_TIMEOUT_SECONDS = 90 # Таймаут ожидания ответа от OpenAI Assistant

# Time Settings
# Загружаем из .env
TIMEZONE_STR = os.getenv("TIMEZONE_STR", "Asia/Yekaterinburg")
WORK_START_HHMM = os.getenv("WORK_START_HHMM", "09:45")
WORK_END_HHMM = os.getenv("WORK_END_HHMM", "19:15")

try:
    TARGET_TZ = pytz.timezone(TIMEZONE_STR)
except pytz.UnknownTimeZoneError:
    logger.error(f"Неизвестный часовой пояс '{TIMEZONE_STR}' в .env. Используется UTC.")
    TARGET_TZ = pytz.utc

def parse_hhmm(time_str: str, default_time: dt_time) -> dt_time:
    """Парсит строку HH:MM в datetime.time, возвращает default_time при ошибке."""
    try:
        hour, minute = map(int, time_str.split(':'))
        # Возвращаем без tzinfo, так как сравнение будет с now_local.time()
        return dt_time(hour, minute)
    except (ValueError, TypeError):
        logger.error(f"Неверный формат времени '{time_str}' в .env. Используется {default_time.strftime('%H:%M')}.")
        return default_time

# Время начала и конца РАБОЧЕГО дня (без tzinfo)
WORK_START_TIME = parse_hhmm(WORK_START_HHMM, dt_time(9, 45))
WORK_END_TIME = parse_hhmm(WORK_END_HHMM, dt_time(19, 15))

# Commands
CMD_SILENCE = "*****"
CMD_SPEAK = "speak"

# Logging Settings
LOGS_DIR = "./logs/context_logs"
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # Можно добавить FileHandler для записи в файл:
        # logging.FileHandler("bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__) # Используем именованный логгер

# --- Validate Configuration ---
required_vars = {
    "VK_GROUP_TOKEN": VK_GROUP_TOKEN,
    "VK_GROUP_ID": VK_GROUP_ID,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "ASSISTANT_ID": ASSISTANT_ID,
    "FOLDER_ID": FOLDER_ID,
    "ADMIN_USER_ID": ADMIN_USER_ID # Проверен выше
}
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"❌ Ошибка: Не найдены переменные в .env: {', '.join(missing_vars)}")

# --- Global State (In-Memory) ---
# ВНИМАНИЕ: Эти данные теряются при перезапуске. Для продакшена рассмотрите БД (Redis, SQLite, etc.)
user_threads: dict[str, str] = {} # {user_key: thread_id}
# user_messages: dict[str, list[dict]] = {} # {user_key: [{'role': '...', 'content': '...', 'timestamp': ...}]}
user_processing_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock) # {peer_id: Lock}
user_last_message_time: dict[int, datetime] = {} # {user_id: timestamp}
chat_silence_state: dict[int, bool] = {} # {peer_id: True if silent}
chat_silence_timers: dict[int, asyncio.Task] = {} # {peer_id: silence timer task}

# Для буферизации сообщений
pending_messages: dict[int, list[str]] = {}  # {peer_id: [text1, text2]} - Буфер сообщений
user_message_timers: dict[int, asyncio.Task] = {} # {peer_id: task} - Таймеры для буферизации

# --- Initialize API Clients ---
# OpenAI Client (рекомендуется создавать один клиент)
try:
    openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("Клиент OpenAI инициализирован.")
except Exception as e:
    logger.critical(f"Не удалось инициализировать клиент OpenAI: {e}", exc_info=True)
    sys.exit(1) # Критическая ошибка, выходим

# VK API Session
# ВАЖНО: Как указано выше, стандартный VkBotLongPoll синхронный.
# Для реальной асинхронной работы этот блок нужно переделать под Callback API
# или асинхронную библиотеку VK.
try:
    vk_session_api = vk_api.VkApi(token=VK_GROUP_TOKEN, api_version=VK_API_VERSION)
    # Используем vk_session_api.get_api() для асинхронных вызовов методов API, если нужно
    # vk = vk_session_api.get_api() # Не используется напрямую в async коде ниже, но может пригодиться
    # Асинхронный API клиент (если библиотека поддерживает)
    # В текущей vk_api этого нет в удобном виде для LongPoll
    # vk_async_session = vk_api.async_vk.AsyncVkApi(token=VK_GROUP_TOKEN, api_version=VK_API_VERSION).get_api()
    # Инициализация LongPoll остается синхронной
    # longpoll = VkBotLongPoll(vk_session_api, VK_GROUP_ID) # <-- ЭТА СТРОКА БУДЕТ ЗАКОММЕНТИРОВАНА
    logger.info("VK API сессия инициализирована (СИНХРОННО). Long Poll будет инициализирован в своем потоке.")
except vk_api.AuthError as e:
     logger.critical(f"Ошибка авторизации VK: {e}. Проверьте токен группы.", exc_info=True)
     sys.exit(1)
except Exception as e:
    logger.critical(f"Ошибка инициализации VK API: {e}", exc_info=True)
    sys.exit(1)

# ChromaDB Client
# Глобальная переменная для активной коллекции, будет инициализирована позже
vector_collection: chromadb.api.models.Collection.Collection | None = None

def _get_active_db_subpath() -> str | None:
    """Читает имя активной поддиректории БД из файла ACTIVE_DB_INFO_FILE."""
    try:
        active_db_info_filepath = os.path.join(VECTOR_DB_BASE_PATH, ACTIVE_DB_INFO_FILE)
        if os.path.exists(active_db_info_filepath):
            with open(active_db_info_filepath, "r", encoding="utf-8") as f:
                active_subdir = f.read().strip()
            if active_subdir: # Убедимся, что прочитано не пустое имя
                # Дополнительная проверка, что такая директория существует
                if os.path.isdir(os.path.join(VECTOR_DB_BASE_PATH, active_subdir)):
                    logger.info(f"Найдена активная поддиректория БД: '{active_subdir}'")
                    return active_subdir
                else:
                    logger.warning(f"В файле '{ACTIVE_DB_INFO_FILE}' указана несуществующая поддиректория: '{active_subdir}'")
                    return None    
            else:
                logger.warning(f"Файл '{ACTIVE_DB_INFO_FILE}' пуст.")
                return None
        else:
            logger.info(f"Файл информации об активной БД '{ACTIVE_DB_INFO_FILE}' не найден. База знаний, вероятно, еще не создана.")
            return None
    except Exception as e:
        logger.error(f"Ошибка при чтении файла информации об активной БД: {e}", exc_info=True)
        return None

async def _initialize_active_vector_collection():
    """Инициализирует глобальную vector_collection на основе активной БД."""
    global vector_collection
    active_subdir = _get_active_db_subpath()
    if active_subdir:
        active_db_full_path = os.path.join(VECTOR_DB_BASE_PATH, active_subdir)
        try:
            # Убедимся, что базовая директория существует
            os.makedirs(VECTOR_DB_BASE_PATH, exist_ok=True)
            # Убедимся, что активная директория существует (хотя _get_active_db_subpath уже проверяет)
            os.makedirs(active_db_full_path, exist_ok=True) 
            
            chroma_client_init = chromadb.PersistentClient(path=active_db_full_path)
            vector_collection = chroma_client_init.get_or_create_collection(
                name=VECTOR_DB_COLLECTION_NAME,
                # metadata={
                #     "hnsw:space": "cosine", # Пример настройки индекса (если нужно)
                #     # "hnsw:construction_ef": 200, 
                #     # "hnsw:search_ef": 100 
                # }
            )
            logger.info(f"Успешно подключено к ChromaDB: '{active_db_full_path}'. Коллекция: '{VECTOR_DB_COLLECTION_NAME}'.")
            logger.info(f"Документов в активной коллекции при старте: {vector_collection.count()}")
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB для пути '{active_db_full_path}': {e}. Поиск по базе знаний будет недоступен.", exc_info=True)
            vector_collection = None
    else:
        logger.warning("Не удалось определить активную директорию БД. База знаний будет недоступна.")
        vector_collection = None

# Google Drive Service
def get_drive_service():
    """Получение сервиса Google Drive через сервисный аккаунт"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        logger.info("Сервис Google Drive инициализирован.")
        return service
    except FileNotFoundError:
        logger.error(f"Файл ключа Google Service Account не найден: {SERVICE_ACCOUNT_FILE}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении сервиса Google Drive: {e}", exc_info=True)
        return None

drive_service = get_drive_service() # Инициализируем при старте

# --- Helper Functions ---

def get_user_key(user_id: int) -> str:
    """Преобразует user_id в ключ для словарей состояния."""
    return str(user_id)

def is_non_working_hours() -> bool:
    """Проверяет, является ли текущее время нерабочим в целевом часовом поясе."""
    now_local = datetime.now(TARGET_TZ) # Используем TARGET_TZ из .env
    current_time_local = now_local.time() # Объект datetime.time

    # Сравниваем только время, без tzinfo
    is_non_working = current_time_local >= WORK_END_TIME or current_time_local < WORK_START_TIME

    # Логируем для отладки (опционально)
    # logger.debug(f"Проверка времени: Сейчас {now_local.strftime('%H:%M:%S %Z')}. Нерабочее: {is_non_working}")
    return is_non_working

async def send_vk_message(peer_id: int, message: str):
    """Асинхронная отправка сообщения VK с обработкой ошибок."""
    if not message:
        logger.warning(f"Попытка отправить пустое сообщение в peer_id={peer_id}")
        return
    try:
        # Используем синхронный метод API через to_thread
        await asyncio.to_thread(
            vk_session_api.method, # Передаем сам метод
            'messages.send',
            {
                'peer_id': peer_id,
                'message': message,
                'random_id': vk_api.utils.get_random_id() # Важно для предотвращения дублей
            }
        )
        # logger.info(f"Сообщение отправлено в peer_id={peer_id}: '{message[:50]}...'")
    except vk_api.exceptions.ApiError as e:
        logger.error(f"Ошибка VK API при отправке сообщения в peer_id={peer_id}: {e}", exc_info=True)
        # Здесь можно добавить обработку специфических ошибок (например, пользователь заблокировал)
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при отправке сообщения в peer_id={peer_id}: {e}", exc_info=True)

async def set_typing_activity(peer_id: int):
     """Устанавливает статус 'печатает' в VK."""
     try:
        await asyncio.to_thread(
             vk_session_api.method,
             'messages.setActivity',
             {'type': 'typing', 'peer_id': peer_id}
         )
     except Exception as e:
         logger.warning(f"Не удалось установить статус 'typing' для peer_id={peer_id}: {e}")


# --- Silence Mode Management ---

async def silence_timeout_task(peer_id: int, timeout_seconds: int):
    """Асинхронная задача, ожидающая таймаут и снимающая режим молчания."""
    logger.debug(f"Запущен таймер молчания на {timeout_seconds}с для peer_id={peer_id}")
    try:
        await asyncio.sleep(timeout_seconds)
        # Проверяем, не был ли статус изменен вручную за время ожидания
        if chat_silence_state.get(peer_id, False):
            logger.info(f"Таймаут молчания ({timeout_seconds}с) истек для peer_id={peer_id}. Снимаем режим молчания.")
            await unsilence_user(peer_id, is_auto=True) # is_auto, чтобы избежать лишнего логирования отмены таймера
    except asyncio.CancelledError:
        logger.info(f"Таймер молчания для peer_id={peer_id} был отменен.")
    except Exception as e:
        logger.error(f"Ошибка в задаче таймера молчания для peer_id={peer_id}: {e}", exc_info=True)
    finally:
        # Убираем завершенную или отмененную задачу из словаря, если она там еще есть
        current_task = chat_silence_timers.pop(peer_id, None)
        if current_task:
            logger.debug(f"Задача таймера удалена из chat_silence_timers для peer_id={peer_id}")

async def silence_user(peer_id: int):
    """Активирует режим молчания для пользователя/чата."""
    logger.info(f"Активация режима молчания для peer_id={peer_id} на {MANAGER_ACTIVE_TIMEOUT_SECONDS} секунд.")
    chat_silence_state[peer_id] = True

    # Отменяем предыдущий таймер, если он существует
    existing_task = chat_silence_timers.pop(peer_id, None)
    if existing_task:
        try:
            existing_task.cancel()
            await asyncio.sleep(0) # Даем циклу событий обработать отмену
            logger.debug(f"Предыдущий таймер для peer_id={peer_id} отменен.")
        except Exception as e:
            logger.warning(f"Не удалось отменить предыдущий таймер для peer_id={peer_id}: {e}")

    # Создаем и сохраняем новую задачу таймера
    timer_task = asyncio.create_task(silence_timeout_task(peer_id, MANAGER_ACTIVE_TIMEOUT_SECONDS))
    chat_silence_timers[peer_id] = timer_task
    logger.info(f"Запущен таймер молчания для peer_id={peer_id}")

async def unsilence_user(peer_id: int, is_auto: bool = False):
    """Деактивирует режим молчания для пользователя/чата."""
    if chat_silence_state.get(peer_id, False):
        log_prefix = "Автоматическая деактивация" if is_auto else "Ручная деактивация"
        logger.info(f"{log_prefix} режима молчания для peer_id={peer_id}.")
        chat_silence_state[peer_id] = False # Используем False вместо удаления для явности

        # Отменяем активный таймер, если он есть
        existing_task = chat_silence_timers.pop(peer_id, None)
        if existing_task:
            try:
                existing_task.cancel()
                await asyncio.sleep(0)
                logger.info(f"Таймер молчания для peer_id={peer_id} отменен.")
            except Exception as e:
                 logger.warning(f"Не удалось отменить таймер при снятии молчания для peer_id={peer_id}: {e}")
    else:
         logger.debug(f"Попытка снять молчание для peer_id={peer_id}, но бот и так был активен.")

# --- Функции буферизации сообщений ---

async def schedule_buffered_processing(peer_id: int, original_user_id: int):
    """
    Задача, которая ждет MESSAGE_BUFFER_SECONDS и затем вызывает process_buffered_messages.
    """
    log_prefix = f"schedule_buffered_processing(peer:{peer_id}, user:{original_user_id}):"
    current_task = asyncio.current_task() # Получаем текущую задачу для сравнения
    try:
        logger.debug(f"{log_prefix} Ожидание {MESSAGE_BUFFER_SECONDS} секунд...")
        await asyncio.sleep(MESSAGE_BUFFER_SECONDS)

        task_in_dict = user_message_timers.get(peer_id)

        # Таймер сработал. Проверяем, актуальна ли эта задача таймера.
        # Это важно, так как новое сообщение могло создать новый таймер, отменив старый.
        if task_in_dict is not current_task:
            logger.info(f"{log_prefix} Таймер сработал, но он устарел (был заменен новым или отменен). Обработка отменена.")
            return

        # Если это все еще актуальный таймер (не был отменен и заменен), удаляем его и запускаем обработку.
        # Удаляем из словаря ДО вызова process_buffered_messages, чтобы избежать гонки состояний.
        if peer_id in user_message_timers:
            del user_message_timers[peer_id]
        
        logger.debug(f"{log_prefix} Таймер сработал и удален. Вызов process_buffered_messages.")
        asyncio.create_task(process_buffered_messages(peer_id, original_user_id))

    except asyncio.CancelledError:
        logger.info(f"{log_prefix} Таймер отменен (вероятно, пришло новое сообщение или команда).")
        # Если таймер был отменен, он уже должен быть удален из словаря 
        # (при создании нового таймера или при явной отмене).
        # Дополнительная проверка не помешает, но обычно не требуется.
    except Exception as e:
        logger.error(f"{log_prefix} Ошибка в задаче таймера: {e}", exc_info=True)
        # В случае другой ошибки, если таймер все еще принадлежит этой задаче, удаляем его.
        if peer_id in user_message_timers and user_message_timers.get(peer_id) is current_task:
            del user_message_timers[peer_id]

async def process_buffered_messages(peer_id: int, original_user_id: int):
    """
    Обрабатывает накопленные сообщения для пользователя после срабатывания таймера.
    Использует существующую блокировку user_processing_locks[peer_id].
    """
    log_prefix = f"process_buffered_messages(peer:{peer_id}, user:{original_user_id}):"
    logger.debug(f"{log_prefix} Начало обработки буферизованных сообщений.")

    # Блокировка для предотвращения параллельной обработки от одного пользователя (peer_id)
    async with user_processing_locks[peer_id]:
        logger.debug(f"{log_prefix} Блокировка для peer_id={peer_id} получена.")
        
        # Забираем сообщения из буфера. Pop удалит ключ, если список станет пустым.
        messages_to_process = pending_messages.pop(peer_id, [])

        # На всякий случай, если таймер не был удален в schedule_buffered_processing
        # (маловероятно при корректной работе, но для надежности)
        if peer_id in user_message_timers:
            logger.warning(f"{log_prefix} Таймер для peer_id={peer_id} все еще существовал в user_message_timers! Отменяем и удаляем.")
            timer_to_cancel = user_message_timers.pop(peer_id) # Удаляем
            if not timer_to_cancel.done():
                try:
                    timer_to_cancel.cancel()
                except Exception as e_inner_cancel:
                    logger.debug(f"{log_prefix} Ошибка при попытке отменить \"бродячий\" таймер: {e_inner_cancel}")

        if not messages_to_process:
            logger.info(f"{log_prefix} Нет сообщений в буфере для peer_id={peer_id}. Обработка не требуется.")
            return # Блокировка будет освобождена

        combined_input = "\n".join(messages_to_process)
        num_messages = len(messages_to_process)
        logger.info(f'{log_prefix} Объединенный запрос для peer_id={peer_id} ({num_messages} сообщ.): "{combined_input[:200]}..."')

        try:
            # Устанавливаем статус "печатает"
            await set_typing_activity(peer_id)

            # Получаем ответ от ассистента (используем original_user_id для логики OpenAI)
            response_text = await chat_with_assistant(original_user_id, combined_input)

            # Отправляем ответ пользователю (в peer_id)
            await send_vk_message(peer_id, response_text)
            logger.info(f"{log_prefix} Успешно обработан и отправлен ответ на объединенное сообщение для peer_id={peer_id}.")

        except Exception as e:
            logger.error(f"{log_prefix} Ошибка при обработке объединенного сообщения или отправке ответа для peer_id={peer_id}: {e}", exc_info=True)
            try:
                await send_vk_message(peer_id, "Произошла внутренняя ошибка при обработке вашего запроса. Попробуйте позже.")
            except Exception as send_err_e:
                logger.error(f"{log_prefix} Не удалось отправить сообщение об ошибке пользователю peer_id={peer_id}: {send_err_e}")
        finally:
            logger.debug(f"{log_prefix} Блокировка для peer_id={peer_id} освобождена.")

# --- OpenAI Assistant Interaction ---

async def get_or_create_thread(user_id: int) -> str | None:
    """Получает или создает новый thread_id для пользователя."""
    user_key = get_user_key(user_id)
    if user_key in user_threads:
        thread_id = user_threads[user_key]
        try:
            # Легкая проверка существования треда
            await openai_client.beta.threads.messages.list(
                thread_id=thread_id,
                limit=1
            )
            logger.info(f"Используем существующий тред {thread_id} для user_id={user_id}")
            return thread_id
        except openai.NotFoundError:
            logger.warning(f"Тред {thread_id} не найден в OpenAI для user_id={user_id}. Создаем новый.")
        except Exception as e:
            logger.error(f"Ошибка доступа к треду {thread_id} для user_id={user_id}: {e}. Создаем новый.")
            # Удаляем невалидный ID из кэша
            del user_threads[user_key]

    # Создаем новый тред
    try:
        logger.info(f"Создаем новый тред для user_id={user_id}...")
        thread = await openai_client.beta.threads.create()
        thread_id = thread.id
        user_threads[user_key] = thread_id
        logger.info(f"Создан новый тред {thread_id} для user_id={user_id}")
        return thread_id
    except Exception as e:
        logger.error(f"Критическая ошибка при создании нового треда для user_id={user_id}: {e}", exc_info=True)
        return None

async def chat_with_assistant(user_id: int, message_text: str) -> str:
    """Отправляет сообщение ассистенту OpenAI и получает ответ."""
    user_key = get_user_key(user_id)
    thread_id = await get_or_create_thread(user_id)
    if not thread_id:
        return "Произошла внутренняя ошибка при обработке вашего запроса." # ИЗМЕНЕНО

    try:
        # 1. Получаем релевантный контекст из базы знаний
        context = ""
        if vector_collection: # Если база знаний доступна
             context = await get_relevant_context(message_text, k=RELEVANT_CONTEXT_COUNT)
             await log_context(user_id, message_text, context) # Логируем найденный контекст

        # 2. Формируем полный промпт (контекст + вопрос)
        full_prompt = message_text
        if context:
            # Добавляем явно, чтобы ассистент понимал источник
            full_prompt = f"Используй следующую информацию из базы знаний для ответа:\n--- НАЧАЛО КОНТЕКСТА ---\n{context}\n--- КОНЕЦ КОНТЕКСТА ---\n\nВопрос пользователя: {message_text}"
        else:
            logger.info(f"Контекст для запроса user_id={user_id} не найден или база знаний отключена.")

        # 3. Проверяем и отменяем активные запуски для этого треда
        try:
            runs = await openai_client.beta.threads.runs.list(thread_id=thread_id)
            active_runs = [run for run in runs.data if run.status in ['queued', 'in_progress', 'requires_action']]
            if active_runs:
                logger.warning(f"Найдены активные запуски ({len(active_runs)}) для треда {thread_id}. Отменяем...")
                for run in active_runs:
                    try:
                        await openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                        logger.info(f"Отменен активный запуск {run.id} для треда {thread_id}")
                    except Exception as cancel_error:
                        logger.warning(f"Не удалось отменить запуск {run.id}: {cancel_error}")
        except Exception as list_runs_error:
            logger.warning(f"Ошибка при проверке активных запусков для треда {thread_id}: {list_runs_error}")

        # 4. Добавляем сообщение пользователя в тред
        await openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=full_prompt # Отправляем промпт с контекстом
        )
        logger.info(f"Сообщение добавлено в тред {thread_id} для user_id={user_id}")
        # Добавляем в локальную историю (если нужно)
        # await add_message_to_history(user_id, "user", message_text) # Чистый текст вопроса

        # 5. Запускаем ассистента
        run = await openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
            # Можно передать доп. инструкции здесь, если нужно:
            # instructions="Отвечай кратко и по делу."
        )
        logger.info(f"Запущен новый run {run.id} для треда {thread_id}")

        # 6. Ожидаем завершения run с таймаутом
        start_time = time_module.time() # Используем time_module.time() вместо time.time()
        while time_module.time() - start_time < OPENAI_RUN_TIMEOUT_SECONDS: # И здесь тоже
            await asyncio.sleep(1) # Пауза перед проверкой статуса
            run_status = await openai_client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            # logger.debug(f"Run {run.id} status: {run_status.status}")

            if run_status.status == 'completed':
                logger.info(f"Run {run.id} успешно завершен.")
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                error_message = f"Run {run.id} завершился со статусом '{run_status.status}'."
                last_error = getattr(run_status, 'last_error', None)
                if last_error: error_message += f" Ошибка: {last_error.message} (Код: {last_error.code})"
                logger.error(error_message)
                return "Произошла внутренняя ошибка при обработке вашего запроса." # ИЗМЕНЕНО
            elif run_status.status == 'requires_action':
                 logger.warning(f"Run {run.id} требует действия (Function Calling?), что не поддерживается в текущей конфигурации.")
                 # TODO: Обработать Function Calling, если он настроен у ассистента
                 # Пока просто прерываем как ошибку
                 await openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                 return "Произошла внутренняя ошибка при обработке вашего запроса." # ИЗМЕНЕНО

        else: # Сработал таймаут цикла while
            logger.warning(f"Превышено время ожидания ({OPENAI_RUN_TIMEOUT_SECONDS}s) ответа от ассистента для run {run.id}, тред {thread_id}")
            try:
                await openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                logger.info(f"Отменен run {run.id} из-за таймаута.")
            except Exception as cancel_error:
                logger.warning(f"Не удалось отменить run {run.id} после таймаута: {cancel_error}")
            return "Произошла внутренняя ошибка при обработке вашего запроса." # ИЗМЕНЕНО

        # 7. Получаем ответ ассистента
        messages_response = await openai_client.beta.threads.messages.list(
            thread_id=thread_id,
            order="desc", # Новые сообщения сначала
            limit=5 # Берем несколько последних на случай системных сообщений
        )

        assistant_response = None
        for msg in messages_response.data:
            if msg.role == "assistant" and msg.run_id == run.id: # Ищем ответ именно этого run
                if msg.content and msg.content[0].type == 'text':
                    assistant_response = msg.content[0].text.value
                    logger.info(f"Получен ответ от ассистента для user_id={user_id}: {assistant_response[:100]}...")
                    break # Нашли нужный ответ

        if assistant_response:
            # await add_message_to_history(user_id, "assistant", assistant_response) # Добавляем в локальную историю
            await log_context(user_id, message_text, context, assistant_response) # Логируем ответ
            return assistant_response
        else:
            logger.warning(f"Не найдено текстового ответа от ассистента в треде {thread_id} после run {run.id}. Ответы: {messages_response.data}")
            # Пытаемся найти хоть какой-то ответ ассистента, даже если run_id не совпал (на случай глюков API)
            for msg in messages_response.data:
                 if msg.role == "assistant":
                      if msg.content and msg.content[0].type == 'text':
                           logger.warning(f"Найден ответ ассистента, но от другого run ({msg.run_id}) - используем его.")
                           return msg.content[0].text.value
            return "Произошла внутренняя ошибка при обработке вашего запроса." # ИЗМЕНЕНО

    except openai.APIError as e:
         logger.error(f"OpenAI API ошибка для user_id={user_id}: {e}", exc_info=True)
         return "Произошла внутренняя ошибка при обработке вашего запроса." # ИЗМЕНЕНО
    except Exception as e:
        logger.error(f"Непредвиденная ошибка в chat_with_assistant для user_id={user_id}: {e}", exc_info=True)
        return "Произошла внутренняя ошибка при обработке вашего запроса." # Оставлено как есть


# --- Vector Store Management (ChromaDB) ---

async def get_relevant_context(query: str, k: int) -> str:
    """Получает релевантный контекст из векторного хранилища ChromaDB."""
    if not vector_collection:
        logger.warning("Запрос контекста, но ChromaDB не инициализирована.")
        return "" # Не возвращаем ошибку пользователю, просто работаем без контекста

    try:
        # 1. Создаем эмбеддинг для запроса
        try:
            query_embedding_response = await openai_client.embeddings.create(
                 input=[query],
                 model=EMBEDDING_MODEL,
                 dimensions=EMBEDDING_DIMENSIONS if EMBEDDING_DIMENSIONS else None # OpenAI API требует это для text-embedding-3-*
            )
            query_embedding = query_embedding_response.data[0].embedding
            logger.debug(f"Эмбеддинг для запроса '{query[:50]}...' создан.")
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга запроса: {e}", exc_info=True)
            # Не возвращаем ошибку пользователю, просто работаем без контекста
            return "" # ИЗМЕНЕНО

        # 2. Выполняем поиск в ChromaDB
        try:
            results = await asyncio.to_thread(
                vector_collection.query,
                query_embeddings=[query_embedding],
                n_results=k, # Запрашиваем топ K результатов
                include=["documents", "metadatas", "distances"] # Включаем нужные поля
            )
            logger.debug(f"Поиск в ChromaDB для '{query[:50]}...' выполнен.")
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска в ChromaDB: {e}", exc_info=True)
            return "" # ИЗМЕНЕНО

        # 3. Обрабатываем и форматируем результаты
        if not results or not results.get("ids") or not results["ids"][0]:
            logger.info(f"Релевантных документов не найдено для запроса: '{query[:50]}...'")
            return ""

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        context_pieces = []
        sources_used = set()
        logger.info(f"Найдено {len(documents)} док-в для '{query[:50]}...'. Топ {k}:")
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            source = meta.get('source', 'Неизвестный источник')
            logger.info(f"  #{i+1}: Источник='{source}', Дистанция={dist:.4f}, Контент='{doc[:100]}...'")
            context_piece = f"Из документа '{source}':\n{doc}"
            context_pieces.append(context_piece)
            sources_used.add(source)

        if not context_pieces:
             logger.info(f"Не найдено подходящих фрагментов контекста для '{query[:50]}...'.")
             return ""

        # Сшиваем контекст
        full_context = "\n\n---\n\n".join(context_pieces)
        # Можно добавить информацию об источниках в конец
        # full_context += f"\n\n(Информация из источников: {', '.join(sources_used)})"

        logger.info(f"Сформирован контекст размером {len(full_context)} символов из {len(context_pieces)} фрагментов.")
        return full_context

    except Exception as e:
        logger.error(f"Непредвиденная ошибка при получении контекста: {e}", exc_info=True)
        return "" # ИЗМЕНЕНО

async def update_vector_store():
    """Обновляет векторное хранилище ChromaDB документами из Google Drive."""
    logger.info("--- Запуск обновления базы знаний ---")
    
    previous_active_subpath = _get_active_db_subpath()
    logger.info(f"Предыдущая активная поддиректория (для возможного удаления): '{previous_active_subpath}'")

    os.makedirs(VECTOR_DB_BASE_PATH, exist_ok=True)

    if not drive_service:
        logger.error("Обновление базы знаний невозможно: сервис Google Drive не инициализирован.")
        return {"success": False, "error": "Сервис Google Drive не инициализирован", "added_chunks": 0, "total_chunks": 0}

    timestamp_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_new"
    new_db_path = os.path.join(VECTOR_DB_BASE_PATH, timestamp_dir_name)
    logger.info(f"Создание новой временной директории для БД: {new_db_path}")
    try:
        os.makedirs(new_db_path, exist_ok=True)
    except Exception as e_mkdir:
        logger.error(f"Не удалось создать временную директорию '{new_db_path}': {e_mkdir}. Обновление прервано.", exc_info=True)
        return {"success": False, "error": f"Failed to create temp dir: {e_mkdir}", "added_chunks": 0, "total_chunks": 0}

    temp_vector_collection = None
    try:
        temp_chroma_client = chromadb.PersistentClient(path=new_db_path)
        temp_vector_collection = temp_chroma_client.get_or_create_collection(
            name=VECTOR_DB_COLLECTION_NAME,
        )
        logger.info(f"Временная коллекция '{VECTOR_DB_COLLECTION_NAME}' создана/получена в '{new_db_path}'.")

        logger.info("Получение данных из Google Drive...")
        documents_data = await asyncio.to_thread(read_data_from_drive)
        if not documents_data:
            logger.warning("Не найдено документов в Google Drive или произошла ошибка чтения. Обновление прервано.")
            if new_db_path and os.path.exists(new_db_path):
                try:
                    shutil.rmtree(new_db_path)
                    logger.info(f"Временная директория {new_db_path} удалена, так как не найдено документов.")
                except Exception as e_rm_empty:
                    logger.error(f"Не удалось удалить временную директорию {new_db_path} после отсутствия документов: {e_rm_empty}")
            return {"success": False, "error": "No documents found in Google Drive or read error", "added_chunks": 0, "total_chunks": 0}
        logger.info(f"Получено {len(documents_data)} документов из Google Drive.")

        all_texts = []
        all_metadatas = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""], 
        )
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
            ]
        )
        MD_SECTION_MAX_LEN = 2000 

        logger.info("Разбиение документов на чанки...")
        for doc_info in documents_data:
            doc_name = doc_info['name']
            doc_content = doc_info['content']
            if not doc_content or not doc_content.strip():
                logger.warning(f"Документ '{doc_name}' пуст или содержит только пробелы. Пропускаем.")
                continue

            enhanced_doc_content = f"Документ: {doc_name}\\n\\n{doc_content}"
            chunk_index_in_doc = 0
            
            is_markdown = doc_name.lower().endswith('.md') or \
                          any(marker in doc_content for marker in ['\n# ', '\n## ', '\n### ', '\n#### ', '* ', '- ', '1. '])

            try:
                if is_markdown:
                    logger.info(f"Документ '{doc_name}' обрабатывается как Markdown.")
                    md_header_splits = markdown_splitter.split_text(enhanced_doc_content)
                    
                    for md_split in md_header_splits:
                        headers_metadata = {k: v for k, v in md_split.metadata.items() if k.startswith('header_')}
                        if len(md_split.page_content) > MD_SECTION_MAX_LEN:
                            logger.info(f"Секция Markdown из '{doc_name}' (заголовок: {headers_metadata.get('header_1', 'N/A')}) слишком длинная ({len(md_split.page_content)} символов), разбиваем дополнительно...")
                            smaller_chunks = text_splitter.split_text(md_split.page_content)
                            for sub_chunk in smaller_chunks:
                                all_texts.append(sub_chunk)
                                all_metadatas.append({
                                    "source": doc_name,
                                    "document_type": "markdown_section_split",
                                    **headers_metadata,
                                    "chunk": chunk_index_in_doc
                                })
                                chunk_index_in_doc += 1
                            logger.info(f"Длинная секция разбита на {len(smaller_chunks)} под-чанков.")
                        else:
                            all_texts.append(md_split.page_content)
                            all_metadatas.append({
                                "source": doc_name,
                                "document_type": "markdown",
                                **headers_metadata,
                                "chunk": chunk_index_in_doc
                            })
                            chunk_index_in_doc += 1
                else:
                    logger.info(f"Документ '{doc_name}' обрабатывается как обычный текст.")
                    chunks = text_splitter.split_text(enhanced_doc_content)
                    for chunk_text in chunks:
                        all_texts.append(chunk_text)
                        all_metadatas.append({"source": doc_name, "document_type": "text", "chunk": chunk_index_in_doc})
                        chunk_index_in_doc += 1
                
                logger.info(f"Документ '{doc_name}' разбит на {chunk_index_in_doc} чанков.")

            except Exception as e:
                logger.error(f"Ошибка при разбиении документа '{doc_name}': {e}", exc_info=True)
                if is_markdown:
                    logger.warning(f"Попытка обработать '{doc_name}' как обычный текст после ошибки Markdown-разбиения.")
                    try:
                        chunks = text_splitter.split_text(enhanced_doc_content)
                        chunk_index_in_doc = 0 
                        for chunk_text in chunks:
                            all_texts.append(chunk_text)
                            all_metadatas.append({"source": doc_name, "document_type": "text_fallback", "chunk": chunk_index_in_doc})
                            chunk_index_in_doc += 1
                        logger.info(f"Документ '{doc_name}' (fallback) разбит на {chunk_index_in_doc} чанков.")
                    except Exception as fallback_e:
                        logger.error(f"Ошибка при fallback-разбиении документа '{doc_name}': {fallback_e}", exc_info=True)
                continue 

        if not all_texts:
            logger.warning("После обработки не осталось текстовых данных для добавления в базу. Обновление прервано.")
            if new_db_path and os.path.exists(new_db_path):
                try:
                    shutil.rmtree(new_db_path)
                    logger.info(f"Временная директория {new_db_path} удалена, так как нет данных для добавления.")
                except Exception as e_rm_nodata:
                    logger.error(f"Не удалось удалить временную директорию {new_db_path} после отсутствия данных: {e_rm_nodata}")
            return {"success": False, "error": "No text data left after processing to add to the database", "added_chunks": 0, "total_chunks": 0}

        logger.info(f"Добавление {len(all_texts)} новых чанков во временную коллекцию...")
        try:
            # Генерируем уникальные ID для каждого чанка (важно для ChromaDB)
            # Пример: "source_name_chunk_index"
            all_ids = [f"{meta['source']}_{meta['chunk']}" for meta in all_metadatas]

            # Создаем эмбеддинги (это может занять время и стоить денег!)
            # Используем клиент OpenAI напрямую, так как LangChain Embeddings могут быть синхронными
            logger.info(f"Создание эмбеддингов для {len(all_texts)} чанков с помощью {EMBEDDING_MODEL}...")
            # TODO: Обработка ошибок rate limit от OpenAI (разбиение на батчи, retry)
            embeddings_response = await openai_client.embeddings.create(
                input=all_texts,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS if EMBEDDING_DIMENSIONS else None
            )
            all_embeddings = [item.embedding for item in embeddings_response.data]
            logger.info(f"Эмбеддинги созданы успешно.")

            # Добавляем данные в ChromaDB батчами (Chroma сама разбивает, но лучше контролировать)
            # batch_size = 100 # Пример
            # for i in range(0, len(all_ids), batch_size):
            #    batch_ids = all_ids[i:i+batch_size]
            #    batch_metadatas = all_metadatas[i:i+batch_size]
            #    batch_embeddings = all_embeddings[i:i+batch_size]
            #    batch_documents = all_texts[i:i+batch_size]
            #    await asyncio.to_thread(
            #        vector_collection.add,
            #        ids=batch_ids,
            #        embeddings=batch_embeddings,
            #        metadatas=batch_metadatas,
            #        documents=batch_documents
            #    )
            #    logger.info(f"Добавлена пачка {i//batch_size + 1} / {len(all_ids)//batch_size + 1}")
            #    await asyncio.sleep(0.5) # Небольшая пауза между батчами

            # Или добавляем всё сразу, если чанков не слишком много
            await asyncio.to_thread(
               temp_vector_collection.add, # Добавляем во временную коллекцию
               ids=all_ids,
               embeddings=all_embeddings,
               metadatas=all_metadatas,
               documents=all_texts # Тексты тоже сохраняем в базе
            )

            logger.info(f"Успешно добавлено {len(all_ids)} чанков во временную коллекцию '{VECTOR_DB_COLLECTION_NAME}'.")
            final_added_chunks = len(all_ids)
            final_total_chunks = temp_vector_collection.count()
            logger.info(f"Текущее количество документов во временной коллекции: {final_total_chunks}")

            # --- УСПЕШНОЕ ЗАВЕРШЕНИЕ ОБНОВЛЕНИЯ ВРЕМЕННОЙ БАЗЫ ---
            # 1. Сохраняем путь к новой активной базе
            active_db_info_filepath = os.path.join(VECTOR_DB_BASE_PATH, ACTIVE_DB_INFO_FILE)
            try:
                with open(active_db_info_filepath, "w", encoding="utf-8") as f:
                    f.write(timestamp_dir_name) # Записываем имя новой активной поддиректории
                logger.info(f"Путь к новой активной базе '{timestamp_dir_name}' сохранен в: {active_db_info_filepath}")
            except Exception as e_save_path:
                logger.error(f"НЕ УДАЛОСЬ сохранить путь к активной базе '{timestamp_dir_name}' в файл {active_db_info_filepath}: {e_save_path}", exc_info=True)
                # Это критично. Хотя временная база создана, бот не сможет ее использовать.
                # Удаляем временную базу, чтобы избежать мусора и ложных срабатываний в будущем.
                if new_db_path and os.path.exists(new_db_path):
                    shutil.rmtree(new_db_path)
                    logger.info(f"Временная директория {new_db_path} удалена из-за ошибки сохранения пути.")
                return {"success": False, "error": f"DB updated in temp but failed to save active path: {e_save_path}", "added_chunks": final_added_chunks, "total_chunks": final_total_chunks}

            # 2. Перезагружаем глобальную vector_collection, чтобы она указывала на новую базу
            logger.info("Перезагрузка глобальной vector_collection для использования новой активной базы...")
            await _initialize_active_vector_collection() # Это обновит глобальную vector_collection
            if not vector_collection:
                 logger.error("Критическая ошибка: не удалось перезагрузить vector_collection на новую активную базу после обновления!")
                 # В этом случае, новая база стала активной (путь записан), но бот не может ее использовать.
                 # Это оставит бота без доступа к базе до перезапуска или следующего успешного обновления.
                 return {"success": False, "error": "Failed to reload global vector_collection to new active DB", "added_chunks": final_added_chunks, "total_chunks": final_total_chunks} 

            # 3. Удаляем предыдущую активную директорию (если она была и отличается от новой)
            if previous_active_subpath and previous_active_subpath != timestamp_dir_name:
                previous_active_full_path = os.path.join(VECTOR_DB_BASE_PATH, previous_active_subpath)
                if os.path.exists(previous_active_full_path):
                    try:
                        shutil.rmtree(previous_active_full_path)
                        logger.info(f"Успешно удалена предыдущая активная директория БД: '{previous_active_full_path}'")
                    except Exception as e_rm_old:
                        logger.error(f"Не удалось удалить предыдущую активную директорию БД '{previous_active_full_path}': {e_rm_old}", exc_info=True)
                else:
                    logger.warning(f"Предыдущая активная директория '{previous_active_full_path}' не найдена для удаления (возможно, уже удалена или это первый запуск).")
            else:
                if previous_active_subpath:
                     logger.info("Новая активная директория совпадает с предыдущей. Удаление не требуется.")
                else:
                     logger.info("Предыдущей активной директории не было. Удаление не требуется.")

            logger.info("--- Обновление базы знаний успешно завершено ---")
            return {"success": True, "added_chunks": final_added_chunks, "total_chunks": final_total_chunks, "new_active_path": timestamp_dir_name}

        except openai.APIError as e:
             logger.error(f"OpenAI API ошибка при создании эмбеддингов: {e}", exc_info=True)
             # Удаляем временную базу при ошибке OpenAI
             if new_db_path and os.path.exists(new_db_path):
                 shutil.rmtree(new_db_path)
                 logger.info(f"Временная директория {new_db_path} удалена из-за ошибки OpenAI.")
             return {"success": False, "error": f"OpenAI API error: {e}", "added_chunks": 0, "total_chunks": 0}
        except Exception as e:
            logger.error(f"Ошибка при добавлении данных в ChromaDB во временной директории: {e}", exc_info=True)
            # Удаляем временную базу при другой ошибке
            if new_db_path and os.path.exists(new_db_path):
                shutil.rmtree(new_db_path)
                logger.info(f"Временная директория {new_db_path} удалена из-за ошибки добавления данных.")
            return {"success": False, "error": f"ChromaDB add error: {e}", "added_chunks": 0, "total_chunks": 0}

    except Exception as e:
        logger.error(f"Критическая ошибка во время обновления базы знаний: {e}", exc_info=True)
        # Удаляем временную базу, если она была создана и произошла общая ошибка
        if new_db_path and os.path.exists(new_db_path):
            try:
                shutil.rmtree(new_db_path)
                logger.info(f"Временная директория {new_db_path} удалена из-за общей ошибки обновления.")
            except Exception as e_rm_final:
                logger.error(f"Не удалось удалить временную директорию {new_db_path} после общей ошибки: {e_rm_final}")
        return {"success": False, "error": f"Critical update error: {e}", "added_chunks": 0, "total_chunks": 0}


# --- Google Drive Reading ---

def read_data_from_drive() -> list[dict]:
    """Читает текстовое содержимое поддерживаемых файлов из Google Drive."""
    if not drive_service:
        logger.error("Чтение из Google Drive невозможно: сервис не инициализирован.")
        return []

    result_docs = []
    try:
        logger.info(f"Запрос списка файлов из папки Google Drive ID: {FOLDER_ID}")
        files_response = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and trashed=false", # Не читаем из корзины
            fields="files(id, name, mimeType)",
            pageSize=1000 # Увеличиваем лимит
        ).execute()

        files = files_response.get('files', [])
        logger.info(f"Найдено {len(files)} файлов в папке.")

        downloader_map = {
            'application/vnd.google-apps.document': download_google_doc,
            'application/pdf': download_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': download_docx,
            'text/plain': download_text,
            # Можно добавить другие типы, например, 'text/markdown'
        }

        for file in files:
            file_id = file['id']
            mime_type = file['mimeType']
            file_name = file['name']

            if mime_type in downloader_map:
                logger.info(f"Обработка файла: '{file_name}' (ID: {file_id}, Type: {mime_type})")
                downloader_func = downloader_map[mime_type]
                try:
                    # Синхронный вызов функции скачивания
                    content = downloader_func(drive_service, file_id)
                    if content and content.strip():
                        result_docs.append({'name': file_name, 'content': content})
                        logger.info(f"Успешно прочитан файл: '{file_name}' ({len(content)} символов)")
                    else:
                        logger.warning(f"Файл '{file_name}' пуст или не удалось извлечь контент.")
                except Exception as e:
                    logger.error(f"Ошибка чтения файла '{file_name}' (ID: {file_id}): {e}", exc_info=True)
            else:
                logger.debug(f"Файл '{file_name}' имеет неподдерживаемый тип ({mime_type}). Пропускаем.")

    except Exception as e:
        logger.error(f"Критическая ошибка при чтении из Google Drive: {e}", exc_info=True)
        return [] # Возвращаем пустой список при ошибке

    logger.info(f"Чтение из Google Drive завершено. Прочитано {len(result_docs)} документов.")
    return result_docs

def download_google_doc(service, file_id):
    """Скачивает и читает содержимое Google Doc как plain text."""
    request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        # logger.debug(f"Download Google Doc {file_id}: {int(status.progress() * 100)}%.")
    return fh.getvalue().decode('utf-8', errors='ignore') # Игнорируем ошибки декодирования

def download_pdf(service, file_id):
    """Скачивает и читает текст из PDF файла."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    try:
        pdf_reader = PyPDF2.PdfReader(fh)
        text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
        return text
    except PyPDF2.errors.PdfReadError as e:
         logger.warning(f"Не удалось прочитать PDF файл (ID: {file_id}): {e}. Возможно, зашифрован или поврежден.")
         return ""
    except Exception as e:
         logger.error(f"Ошибка обработки PDF (ID: {file_id}): {e}", exc_info=True)
         return ""


def download_docx(service, file_id):
    """Скачивает и читает текст из DOCX файла."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    try:
        doc = docx.Document(fh)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
    except Exception as e: # Ловим общую ошибку, т.к. python-docx может кидать разные
         logger.error(f"Ошибка обработки DOCX (ID: {file_id}): {e}", exc_info=True)
         return ""


def download_text(service, file_id):
    """Скачивает и читает текстовый файл."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    # Пытаемся определить кодировку (простой вариант)
    try:
        return fh.getvalue().decode('utf-8')
    except UnicodeDecodeError:
         logger.warning(f"Не удалось декодировать текстовый файл {file_id} как UTF-8, пробуем cp1251.")
         try:
              return fh.getvalue().decode('cp1251', errors='ignore')
         except Exception as e:
              logger.error(f"Не удалось декодировать текстовый файл {file_id} ни в одной из кодировок: {e}")
              return ""


# --- History and Context Management ---

# async def add_message_to_history(user_id: int, role: str, content: str):
#     """Добавляет сообщение в локальную историю пользователя (в памяти)."""
#     user_key = get_user_key(user_id)
#     if user_key not in user_messages:
#         user_messages[user_key] = []
# 
#     # Ограничиваем длину для экономии памяти
#     max_len = 1500
#     truncated_content = content[:max_len] + ('...' if len(content) > max_len else '')
# 
#     user_messages[user_key].append({
#         'role': role,
#         'content': truncated_content,
#         'timestamp': datetime.now()
#     })
#     # Очистка старых сообщений при добавлении (можно вынести в фоновую задачу)
#     await cleanup_old_messages_for_user(user_key)
# 
# 
# async def cleanup_old_messages_for_user(user_key: str):
#      """Удаляет сообщения старше MESSAGE_LIFETIME_DAYS для конкретного пользователя."""
#      if user_key in user_messages:
#         now_dt = datetime.now() # Используем импортированный класс datetime
#         lifetime = timedelta(days=MESSAGE_LIFETIME_DAYS)
#         original_count = len(user_messages[user_key])
#         user_messages[user_key] = [
#             msg for msg in user_messages[user_key]
#             if now_dt - msg['timestamp'] < lifetime
#         ]
#         removed_count = original_count - len(user_messages[user_key])
#         if removed_count > 0:
#             logger.info(f"Очищена история для user_key={user_key}. Удалено {removed_count} старых сообщений.")
# 
# 
# async def cleanup_all_old_messages():
#     """Очищает старые сообщения для всех пользователей."""
#     logger.info("Запуск очистки старых сообщений в памяти...")
#     # Итерируемся по копии ключей, так как словарь может изменяться
#     for user_key in list(user_messages.keys()):
#         await cleanup_old_messages_for_user(user_key)
#     logger.info("Очистка старых сообщений завершена.")


async def log_context(user_id, message_text, context, response_text=None):
    """Логирует контекст, запрос и (опционально) ответ в файл."""
    try:
        ts = datetime.now()
        log_filename = os.path.join(LOGS_DIR, f"context_{user_id}_{ts.strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {ts.isoformat()}\n")
            f.write(f"User ID: {user_id}\n")
            f.write("="*20 + " User Query " + "="*20 + "\n")
            f.write(message_text + "\n")
            f.write("="*20 + " Retrieved Context " + "="*20 + "\n")
            f.write(context if context else "Контекст не найден или не использовался.\n")
            if response_text:
                f.write("="*20 + " Assistant Response " + "="*20 + "\n")
                f.write(response_text + "\n")
            f.write("="*50 + "\n")
        # logger.debug(f"Контекст сохранен в {log_filename}")
    except Exception as e:
        logger.error(f"Ошибка при логировании контекста для user_id={user_id}: {e}", exc_info=True)


async def cleanup_old_context_logs():
    """Удаляет файлы логов контекста старше LOG_RETENTION_SECONDS."""
    logger.info("Запуск очистки старых логов контекста...")
    count = 0
    try:
        now = time_module.time() # Используем time_module.time()
        cutoff = now - LOG_RETENTION_SECONDS
        for filename in glob.glob(os.path.join(LOGS_DIR, "context_*.log")):
            try:
                file_mod_time = os.path.getmtime(filename)
                if file_mod_time < cutoff:
                    os.remove(filename)
                    count += 1
            except FileNotFoundError:
                continue # Файл уже удален
            except Exception as e:
                logger.error(f"Ошибка при удалении файла лога {filename}: {e}")

        if count > 0:
            logger.info(f"Очистка логов контекста: удалено {count} устаревших файлов.")
        else:
            logger.info("Очистка логов контекста: устаревших файлов не найдено.")
    except Exception as e:
        logger.error(f"Критическая ошибка при очистке логов контекста: {e}", exc_info=True)


# --- Background Cleanup Task ---

# Переменная для отслеживания даты последнего автообновления БЗ
last_auto_update_date: Optional[datetime.date] = None 

async def background_cleanup_task():
    """Периодически запускает задачи очистки и ежедневное обновление БЗ."""
    global last_auto_update_date
    while True:
        await asyncio.sleep(3600) # Проверять каждый час
        logger.info("Запуск периодической фоновой задачи...")
        
        # --- Ежедневное обновление базы знаний в 4 утра ---
        try:
            now_local = datetime.now(TARGET_TZ) # Получаем текущее время в целевом часовом поясе
            # logger.debug(f"Фоновая задача: сейчас {now_local.strftime('%Y-%m-%d %H:%M')} в {TARGET_TZ}")

            # Проверяем, что сейчас 4 часа утра и что сегодня обновление еще не выполнялось
            if now_local.hour == 4 and (last_auto_update_date is None or last_auto_update_date < now_local.date()):
                logger.info(f"Время для ежедневного обновления базы знаний ({now_local.hour}:00). Запускаем...")
                await run_update_and_notify_admin(ADMIN_USER_ID) # Запускаем обновление и уведомление
                last_auto_update_date = now_local.date() # Обновляем дату последнего запуска
                logger.info(f"Ежедневное обновление БЗ за {last_auto_update_date} выполнено (или запущено).")
            # else:
                # logger.debug(f"Фоновая задача: Условие для автообновления БЗ не выполнено (час: {now_local.hour}, дата последнего обновления: {last_auto_update_date})")
        except Exception as e_auto_update:
            logger.error(f"Ошибка в логике ежедневного обновления БЗ: {e_auto_update}", exc_info=True)
        # --------------------------------------------------

        # await cleanup_all_old_messages() # Если эта функция когда-то была, она закомментирована
        await cleanup_old_context_logs()
        logger.info("Фоновая задача: очистка старых логов контекста завершена.")
        # Можно добавить очистку других ресурсов, если нужно
        logger.info("Периодическая фоновая задача завершила цикл.")


# --- Main Event Handler ---

async def handle_new_message(event: VkBotEvent):
    """Обрабатывает новое входящее сообщение."""
    # Явно указываем, что будем использовать глобальные переменные
    global user_threads # Убираем pending_messages, user_message_timers, т.к. они теперь глобальные
    
    try:
        if event.from_user: # Обрабатываем только личные сообщения боту
            user_id = event.obj.message['from_id']
            peer_id = event.obj.message['peer_id'] # ИЗМЕНЕНО: Получаем peer_id из объекта сообщения
            message_text = event.obj.message['text'].strip()
            # message_id = event.obj.message['id'] # ID сообщения VK (может пригодиться)

            if not message_text: # Игнорируем пустые сообщения
                 logger.info(f"Получено пустое сообщение от user_id={user_id}. Игнорируем.")
                 return

            # --- Команда обновления базы знаний (только для администратора) ---
            if message_text.lower() == "/update" and user_id == ADMIN_USER_ID:
                logger.info(f"Администратор {user_id} (peer_id={peer_id}) инициировал обновление базы знаний командой /update.")
                await send_vk_message(peer_id, "🔄 Запускаю обновление базы знаний. Это может занять некоторое время... О результате сообщу.")
                # Запускаем обновление в фоне, чтобы не блокировать обработку других команд/сообщений
                asyncio.create_task(run_update_and_notify_admin(peer_id)) 
                return # Команда обработана
            # --- Конец команды /update ---

            # --- НОВОЕ: Проверка команды /reset ---            
            if message_text.lower() == "/reset":
                user_key = get_user_key(user_id)
                log_prefix = f"handle_new_message(reset for peer:{peer_id}, user:{user_id}):"
                logging.info(f"{log_prefix} Получена команда сброса диалога.")

                # 1. Очистка буфера и таймера
                if peer_id in pending_messages:
                    del pending_messages[peer_id]
                    logging.debug(f"{log_prefix} Буфер сообщений очищен.")
                if peer_id in user_message_timers:
                    old_timer = user_message_timers.pop(peer_id)
                    if not old_timer.done():
                        try:
                            old_timer.cancel()
                            logging.debug(f"{log_prefix} Таймер обработки отменен.")
                        except Exception as e:
                            logging.warning(f"{log_prefix} Ошибка при отмене таймера: {e}")

                # 2. Очистка локальной истории
                # if user_key in user_messages:
                #     del user_messages[user_key]
                #     logging.debug(f"{log_prefix} Локальная история сообщений очищена.")

                # 2. Очистка ID треда OpenAI
                thread_id_to_forget = user_threads.pop(user_key, None)
                if thread_id_to_forget:
                    logging.info(f"{log_prefix} Запись о треде {thread_id_to_forget} удалена из памяти бота.")
                    # Опционально: Попытка удалить тред на стороне OpenAI
                    # try:
                    #     logger.debug(f"{log_prefix} Попытка удаления треда {thread_id_to_forget} на OpenAI...")
                    #     await openai_client.beta.threads.delete(thread_id=thread_id_to_forget)
                    #     logging.info(f"{log_prefix} Тред {thread_id_to_forget} успешно удален на OpenAI.")
                    # except Exception as delete_err:
                    #     logging.warning(f"{log_prefix} Не удалось удалить тред {thread_id_to_forget} на OpenAI: {delete_err}")
                else:
                    logging.debug(f"{log_prefix} Не найдено активного треда для сброса.")

                await send_vk_message(peer_id, "🔄 Диалог сброшен. Ваше следующее сообщение начнет новую беседу.")
                return # Завершаем обработку команды
            # --- Конец проверки /reset ---

            # --- НОВОЕ: Проверка команды reset_all (только для администратора) ---
            if message_text.lower() == "/reset_all":
                log_prefix = f"handle_new_message(reset_all from user:{user_id}):"
                logging.info(f"{log_prefix} Получена команда сброса ВСЕХ диалогов.")
                
                # Проверка, что команду вызывает администратор
                if user_id != ADMIN_USER_ID:
                    logging.warning(f"{log_prefix} Отказ: команда доступна только администратору. user_id={user_id} != ADMIN_USER_ID={ADMIN_USER_ID}")
                    await send_vk_message(peer_id, "❌ У вас нет прав для выполнения этой команды. Она доступна только администратору.")
                    return
                
                # Информация о текущем состоянии
                threads_count = len(user_threads)
                pending_count = len(pending_messages)
                timers_count = len(user_message_timers)
                
                logging.warning(f"{log_prefix} Администратор {user_id} инициировал полный сброс всех диалогов!")
                
                # 1. Отмена всех активных таймеров обработки
                active_timer_count = 0
                for task in user_message_timers.values():
                    if not task.done():
                        try:
                            task.cancel()
                            active_timer_count += 1
                        except Exception as e:
                            logging.debug(f"{log_prefix} Ошибка при отмене таймера: {e}")
                user_message_timers.clear()
                logging.info(f"{log_prefix} Отменено {active_timer_count} активных таймеров обработки.")
                
                # 2. Очистка всех буферов
                pending_messages.clear()
                logging.info(f"{log_prefix} Очищены буферы сообщений для {pending_count} пользователей.")
                
                # 3. Очистка всех тредов
                # user_messages.clear() # Закомментировано, так как переменная user_messages не используется
                threads_list = list(user_threads.values())
                user_threads.clear()
                logging.info(f"{log_prefix} Очищены треды {threads_count} пользователей: {threads_list[:5]}{'...' if len(threads_list) > 5 else ''}")
                
                # 4. Отправка подтверждения
                await send_vk_message(peer_id, f"🔄 СБРОС ВСЕХ ДИАЛОГОВ ВЫПОЛНЕН.\n- Отменено таймеров: {active_timer_count}\n- Очищено буферов: {pending_count}\n- Удалено тредов: {threads_count}")
                return # Завершаем обработку команды
            # --- Конец проверки reset_all ---

            is_manager = user_id in MANAGER_USER_IDS or user_id == ADMIN_USER_ID

            # 1. Проверка на команды от менеджера
            if is_manager:
                command = message_text.lower()
                if command == CMD_SILENCE.lower():
                    await silence_user(peer_id)
                    # await send_vk_message(peer_id, "#") # ИЗМЕНЕНО: Закомментирована отправка сообщения
                    return # Команда обработана

                elif command == CMD_SPEAK.lower():
                    await unsilence_user(peer_id)
                    await send_vk_message(peer_id, "🤖 Режим молчания снят. Бот снова активен.")
                    return # Команда обработана
                # Если это не команда, менеджер просто пишет боту - обрабатываем как обычный пользователь

            # 2. Проверка, не находится ли бот в режиме молчания
            if chat_silence_state.get(peer_id, False):
                logger.info(f"Бот в режиме молчания для peer_id={peer_id}. Сообщение '{message_text[:50]}...' от user_id={user_id} игнорируется.")
                return # Игнорируем

            # 2.5. Проверка на рабочее время (ДО кулдауна) - ВРЕМЕННО ОТКЛЮЧЕНО
            # if not is_non_working_hours():
            #     # Логируем текущее время для ясности
            #     now_local_str = datetime.now(TARGET_TZ).strftime('%H:%M:%S %Z') # Используем TARGET_TZ
            #     logger.info(f"Рабочее время ({now_local_str}). Сообщение '{message_text[:50]}...' от user_id={user_id} игнорируется.")
            #     return # Игнорируем сообщение, если сейчас рабочее время

            # 3. Проверка кулдауна
            now_dt = datetime.now() # Используем импортированный класс datetime
            last_time = user_last_message_time.get(user_id)
            if last_time and now_dt - last_time < timedelta(seconds=MESSAGE_COOLDOWN_SECONDS):
                # Отправляем сообщение о кулдауне только если прошло больше N секунд с последнего такого сообщения
                # чтобы не спамить пользователя. Реализация опущена для краткости.
                logger.warning(f"Кулдаун для user_id={user_id}. Игнорируем сообщение.")
                # await send_vk_message(peer_id, f"Пожалуйста, подождите {MESSAGE_COOLDOWN_SECONDS} сек. перед отправкой следующего сообщения.")
                return
            user_last_message_time[user_id] = now_dt

            # 4. Обработка сообщения пользователя (с блокировкой)
            logger.info(f"Получено сообщение от user_id={user_id} (peer_id={peer_id}): '{message_text[:100]}...'")

            # Вместо прямой обработки, добавляем в буфер и управляем таймером
            pending_messages.setdefault(peer_id, []).append(message_text)
            logger.debug(f"Сообщение от peer_id={peer_id} добавлено в буфер. Текущий буфер: {pending_messages[peer_id]}")

            # Отменяем предыдущий таймер для этого peer_id, если он существует и активен
            if peer_id in user_message_timers:
                old_timer = user_message_timers.pop(peer_id) # Удаляем сразу, чтобы избежать гонки
                if not old_timer.done():
                    try:
                        old_timer.cancel()
                        logger.debug(f"Предыдущий таймер для peer_id={peer_id} отменен.")
                    except Exception as e_cancel:
                        logger.warning(f"Не удалось отменить старый таймер для peer_id={peer_id}: {e_cancel}")
            
            # Запускаем новый таймер
            logger.debug(f"Запуск нового таймера буферизации для peer_id={peer_id} ({MESSAGE_BUFFER_SECONDS} сек). Передан user_id={user_id}")
            new_timer_task = asyncio.create_task(schedule_buffered_processing(peer_id, user_id)) # Передаем peer_id и user_id
            user_message_timers[peer_id] = new_timer_task

            # Старая логика прямой обработки теперь будет в process_buffered_messages
            # async with user_processing_locks[peer_id]:
            #     try:
            #         await set_typing_activity(peer_id)
            #         response_text = await chat_with_assistant(user_id, message_text)
            #         await send_vk_message(peer_id, response_text)
            #     except Exception as e:
            #         logger.error(f"Ошибка при обработке сообщения или отправке ответа для user_id={user_id}: {e}", exc_info=True)
            #         await send_vk_message(peer_id, "Произошла внутренняя ошибка при обработке вашего запроса. Попробуйте позже.")
        elif event.from_chat:
            # Логика для групповых чатов (если нужна)
            # Здесь можно было бы реализовать автоматическое замолкание, если пишет менеджер
            chat_id = event.chat_id
            user_id = event.obj.message['from_id']
            message_text = event.obj.message['text'].strip()
            logger.debug(f"Получено сообщение в чате {chat_id} от {user_id}: {message_text[:50]}")
            # TODO: Добавить обработку сообщений из чатов, если бот должен там работать
            pass
        else:
            # Другие типы событий Long Poll
            logger.debug(f"Получено событие Long Poll типа {event.type}, не обрабатывается.")

    except Exception as e:
        # Ловим ошибки на верхнем уровне обработчика событий
        logger.error(f"Критическая ошибка в обработчике событий handle_new_message: {e}", exc_info=True)


# --- Main Application Logic ---

async def run_update_and_notify_admin(notification_peer_id: int):
    """Выполняет обновление базы знаний и уведомляет администратора о результате."""
    logger.info(f"run_update_and_notify_admin: Запуск обновления базы для уведомления peer_id={notification_peer_id}")
    update_result = await update_vector_store()
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    admin_message = f"🔔 Отчет об обновлении базы знаний ({current_time_str}):\n"
    if update_result.get("success"):
        admin_message += f"✅ Успешно обновлено!\n"
        admin_message += f"➕ Добавлено новых чанков: {update_result.get('added_chunks', 'N/A')}\n"
        admin_message += f"📊 Всего чанков в активной базе: {update_result.get('total_chunks', 'N/A')}\n"
        if update_result.get("new_active_path"):
            admin_message += f"📁 Новая активная директория: {update_result['new_active_path']}"
        logger.info(f"Обновление базы знаний завершено успешно. {admin_message}")
    else:
        error_details = update_result.get('error', 'Неизвестная ошибка.')
        admin_message += f"❌ Ошибка при обновлении:\n{error_details}\n"
        admin_message += f"Состояние базы могло не измениться или быть некорректным."
        logger.error(f"Обновление базы знаний завершено с ошибкой: {error_details}")

    try:
        # Отправляем сообщение администратору, если notification_peer_id это его peer_id.
        # Если бот запущен не от имени админа, то это сообщение уйдет в тот чат, откуда пришла команда.
        await send_vk_message(notification_peer_id, admin_message)
        # Дополнительно всегда отправляем администратору в его ЛС, если ADMIN_USER_ID это ID пользователя (а не группы)
        # и если команда была вызвана не из ЛС администратора.
        if ADMIN_USER_ID != 0 and notification_peer_id != ADMIN_USER_ID: # ADMIN_USER_ID может быть id пользователя
             # Проверяем, что ADMIN_USER_ID не является ID группы (обычно они отрицательные или очень большие)
             # Это упрощенная проверка, в реальности ID пользователей и групп могут пересекаться в положительном диапазоне.
             # Для надежности, если ADMIN_USER_ID - это ID пользователя, то он должен быть > 0.
             # Однако, VK API peer_id для пользователя равен user_id.
             if ADMIN_USER_ID > 0: 
                await send_vk_message(ADMIN_USER_ID, "[Автоматическое уведомление] " + admin_message)
    except Exception as e_notify:
        logger.error(f"Не удалось отправить уведомление администратору о результате обновления базы: {e_notify}", exc_info=True)


async def main():
    """Главная асинхронная функция запуска бота."""
    logger.info("--- Запуск VK бота ---")

    # Выполняем первоначальную инициализацию активной коллекции ChromaDB
    await _initialize_active_vector_collection()

    # Запускаем обновление базы знаний при старте (в фоне)
    # Уведомление будет отправлено администратору (ADMIN_USER_ID)
    logger.info("Запуск фонового обновления базы знаний при старте бота...")
    asyncio.create_task(run_update_and_notify_admin(ADMIN_USER_ID))

    # Запуск фоновой задачи очистки
    cleanup_task = asyncio.create_task(background_cleanup_task())
    logger.info("Фоновая задача очистки запущена.")

    # --- ОБРАТИТЕ ВНИМАНИЕ ---
    logger.warning("Используется СИНХРОННЫЙ VkBotLongPoll из стандартной библиотеки vk_api.")
    logger.warning("Это приведет к БЛОКИРОВКЕ асинхронного цикла при ожидании событий.")
    logger.warning("Для корректной асинхронной работы используйте асинхронную библиотеку VK (vkbottle, aiovk)")
    logger.warning("или запустите цикл longpoll.listen() в отдельном потоке (asyncio.to_thread).")
    logger.warning("Текущая реализация представлена для демонстрации структуры.")
    # -------------------------

    listen_task = None
    try:
        # Пример запуска синхронного цикла в отдельном потоке (РЕКОМЕНДУЕТСЯ):
        loop = asyncio.get_running_loop()
        # Оборачиваем to_thread в create_task, чтобы получить объект Task
        listen_task = asyncio.create_task(asyncio.to_thread(run_longpoll_sync, loop), name="VKLongPollListener")

        # ---- ВАРИАНТ НИЖЕ БУДЕТ БЛОКИРОВАТЬ ---
        # logger.info("Запуск цикла прослушивания событий VK Long Poll...")
        # for event in longpoll.listen(): # ЭТО БЛОКИРУЮЩИЙ ВЫЗОВ
        #     if event.type == VkBotEventType.MESSAGE_NEW: # ИЗМЕНЕНО: Закомментировано
        #         # Запускаем обработчик асинхронно, но сам listen() все равно блокирует
        #         asyncio.create_task(handle_new_message(event)) # ИЗМЕНЕНО: Закомментировано
        #     else:
        #         # Обработка других типов событий, если нужно
        #         logger.debug(f"Пропускаем событие типа {event.type}") # ИЗМЕНЕНО: Закомментировано
        # ---- КОНЕЦ БЛОКИРУЮЩЕГО ВАРИАНТА ----

        # Добавляем ожидание завершения задачи прослушивания, если она была запущена
        if listen_task:
             await listen_task

    except vk_api.exceptions.ApiError as e:
        logger.critical(f"Критическая ошибка VK API в цикле Long Poll: {e}", exc_info=True)
    except Exception as e:
         logger.critical(f"Критическая ошибка в главном цикле: {e}", exc_info=True)
    finally:
        logger.info("Завершение работы фоновых задач...")
        cleanup_task.cancel()
        # Отменяем задачи таймеров молчания
        for task in chat_silence_timers.values():
            task.cancel()
        # Если использовался asyncio.to_thread для listen_task, его нужно корректно остановить
        # Прямая отмена потока не всегда возможна, но попытаемся отменить задачу asyncio
        if listen_task and not listen_task.done():
             listen_task.cancel() # Попытка отмены задачи (не остановит поток напрямую)
             logger.warning("Запрошена отмена задачи Long Poll. Сам поток может продолжать работать до следующего события.")

        # Ждем завершения задач (с таймаутом)
        await asyncio.gather(cleanup_task, *chat_silence_timers.values(), listen_task, return_exceptions=True)

        logger.info("--- Бот остановлен ---")


# --- (Опционально) Функция для запуска синхронного LongPoll в потоке ---
def run_longpoll_sync(async_loop: asyncio.AbstractEventLoop):
    logger.info("Запуск синхронного Long Poll в отдельном потоке...")
    
    # Параметры для переподключения
    MAX_RECONNECT_ATTEMPTS = 5  # Максимальное количество попыток подряд
    RECONNECT_DELAY_SECONDS = 30 # Задержка перед повторной попыткой

    current_attempts = 0
    
    # Глобальный vk_session_api и VK_GROUP_ID используются для пересоздания longpoll
    global vk_session_api 
    global VK_GROUP_ID

    while True: # Внешний цикл для переподключения
        try:
            if not vk_session_api: # На случай, если сессия не была создана
                logger.error("[Thread LongPoll] vk_session_api не инициализирована. Невозможно запустить LongPoll.")
                # Можно добавить уведомление администратору или более сложную логику
                time_module.sleep(RECONNECT_DELAY_SECONDS * 5) # Длительная пауза перед следующей попыткой
                continue

            logger.info(f"[Thread LongPoll] Попытка инициализации VkBotLongPoll (попытка {current_attempts + 1}).")
            # Пересоздаем longpoll при каждой попытке подключения (или первой)
            # Это важно, так как предыдущий экземпляр мог быть в невалидном состоянии
            current_longpoll = VkBotLongPoll(vk_session_api, VK_GROUP_ID)
            logger.info("[Thread LongPoll] VkBotLongPoll успешно инициализирован.")
            
            current_attempts = 0 # Сбрасываем счетчик попыток при успешной инициализации

            logger.info("[Thread LongPoll] Начало прослушивания событий...")
            for event in current_longpoll.listen():
                if event.type == VkBotEventType.MESSAGE_NEW:
                    asyncio.run_coroutine_threadsafe(handle_new_message(event), async_loop)
                else:
                    logger.debug(f"[Thread LongPoll] Пропускаем событие типа {event.type}")
            
            # Если цикл listen() завершился штатно (маловероятно для LongPoll, обычно он вечный или падает с ошибкой)
            logger.warning("[Thread LongPoll] Цикл listen() завершился штатно. Перезапуск через задержку...")
            current_attempts = 0 # Сбрасываем, чтобы следующая попытка была как первая
            time_module.sleep(RECONNECT_DELAY_SECONDS)


        except (requests.exceptions.ConnectionError, 
                  requests.exceptions.ReadTimeout, 
                  requests.exceptions.ChunkedEncodingError, # Может возникать при проблемах с сетью
                  vk_api.exceptions.VkApiError, # Общее исключение API VK
                  # BrokenPipeError обычно наследуется от ConnectionError или OSError
                  # OSError # Можно добавить для более широкого охвата
                 ) as e:
            logger.error(f"[Thread LongPoll] Ошибка сети или VK API в цикле Long Poll: {e}", exc_info=True)
            current_attempts += 1
            if MAX_RECONNECT_ATTEMPTS > 0 and current_attempts >= MAX_RECONNECT_ATTEMPTS:
                logger.critical(f"[Thread LongPoll] Превышено максимальное количество ({MAX_RECONNECT_ATTEMPTS}) попыток переподключения. Поток Long Poll останавливается.")
                # Здесь можно добавить отправку уведомления администратору
                asyncio.run_coroutine_threadsafe(send_vk_message(ADMIN_USER_ID, "Критическая ошибка: Поток VK Long Poll остановлен после множества неудачных попыток переподключения."), async_loop)
                break # Выход из внешнего цикла while True

            logger.info(f"[Thread LongPoll] Пауза {RECONNECT_DELAY_SECONDS} секунд перед попыткой {current_attempts + 1}...")
            time_module.sleep(RECONNECT_DELAY_SECONDS)
        
        except Exception as e:
            # Ловим все остальные непредвиденные ошибки, чтобы поток не умер молча
            logger.critical(f"[Thread LongPoll] Непредвиденная критическая ошибка в цикле Long Poll: {e}", exc_info=True)
            # Для таких ошибок можно решить, стоит ли пытаться переподключаться
            # или лучше остановить поток, чтобы не вызывать проблем.
            # Пока что будем пытаться переподключиться, но с увеличенной задержкой.
            current_attempts += 1
            if MAX_RECONNECT_ATTEMPTS > 0 and current_attempts >= MAX_RECONNECT_ATTEMPTS:
                 logger.critical(f"[Thread LongPoll] Превышено максимальное количество ({MAX_RECONNECT_ATTEMPTS}) попыток после непредвиденной ошибки. Поток Long Poll останавливается.")
                 asyncio.run_coroutine_threadsafe(send_vk_message(ADMIN_USER_ID, "Критическая ошибка: Поток VK Long Poll остановлен после множества неудачных попыток (непредвиденная ошибка)."), async_loop)
                 break
            
            logger.info(f"[Thread LongPoll] Пауза {RECONNECT_DELAY_SECONDS * 2} секунд перед попыткой {current_attempts + 1} после непредвиденной ошибки...")
            time_module.sleep(RECONNECT_DELAY_SECONDS * 2)

    logger.info("[Thread LongPoll] Поток Long Poll завершен.")


if __name__ == "__main__":
    try:
        # Python 3.7+
        # asyncio.run(_initialize_active_vector_collection()) # Перенесено в main()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получен сигнал KeyboardInterrupt. Завершение работы...")
    except Exception as e:
         # Ловим ошибки, которые могли возникнуть до запуска asyncio.run()
         logger.critical(f"Критическая ошибка при запуске: {e}", exc_info=True)