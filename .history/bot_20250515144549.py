import sys
import os
import time as time_module
import asyncio
import logging
import datetime
import glob
import io
from io import BytesIO
from collections import defaultdict
from typing import Optional
from datetime import datetime, timedelta
import pytz
import shutil
import requests
import json
import random
import re
import threading

# --- Dependency Imports ---
import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEvent, VkBotEventType
from vk_api.utils import get_random_id

import openai
import chromadb
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import docx
import PyPDF2

# LangChain components for specific tasks
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

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
EMBEDDING_DIMENSIONS = 1536

# Google Drive Settings
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", 'service-account-key.json')
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# User/Manager IDs
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
VECTOR_DB_BASE_PATH = "./local_vector_db"
ACTIVE_DB_INFO_FILE = "active_db_info.txt"
VECTOR_DB_COLLECTION_NAME = "documents_collection"
RELEVANT_CONTEXT_COUNT = 3

# Bot Behavior Settings
MESSAGE_COOLDOWN_SECONDS = 3
MESSAGE_BUFFER_SECONDS = 4
LOG_RETENTION_SECONDS = 86400
OPENAI_RUN_TIMEOUT_SECONDS = 90

# Time Settings
TIMEZONE_STR = os.getenv("TIMEZONE_STR", "Asia/Yekaterinburg")
WORK_START_HHMM = os.getenv("WORK_START_HHMM", "09:45")
WORK_END_HHMM = os.getenv("WORK_END_HHMM", "19:15")

try:
    TARGET_TZ = pytz.timezone(TIMEZONE_STR)
except pytz.UnknownTimeZoneError:
    logging.error(f"Неизвестный часовой пояс '{TIMEZONE_STR}' в .env. Используется UTC.") # Используем logging вместо logger, т.к. logger еще не определен
    TARGET_TZ = pytz.utc

def parse_hhmm(time_str: str, default_time: datetime.time) -> datetime.time:
    try:
        hour, minute = map(int, time_str.split(':'))
        return datetime.time(hour, minute)
    except (ValueError, TypeError):
        logging.error(f"Неверный формат времени '{time_str}' в .env. Используется {default_time.strftime('%H:%M')}.")
        return default_time

WORK_START_TIME = parse_hhmm(WORK_START_HHMM, datetime.time(9, 45))
WORK_END_TIME = parse_hhmm(WORK_END_HHMM, datetime.time(19, 15))

# Commands
CMD_SPEAK = "speak" # Команда для снятия постоянного молчания

# Logging Settings
LOGS_DIR = "./logs/context_logs"
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Validate Configuration ---
required_vars = {
    "VK_GROUP_TOKEN": VK_GROUP_TOKEN,
    "VK_GROUP_ID": VK_GROUP_ID,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "ASSISTANT_ID": ASSISTANT_ID,
    "FOLDER_ID": FOLDER_ID,
    "ADMIN_USER_ID": ADMIN_USER_ID
}
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"❌ Ошибка: Не найдены переменные в .env: {', '.join(missing_vars)}")

# --- Global State (In-Memory) ---
user_threads: dict[str, str] = {}
user_processing_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
user_last_message_time: dict[int, datetime] = {}
chat_silence_state: dict[int, bool] = {} # {peer_id: True if silent by CRM}
MY_PENDING_RANDOM_IDS = set()

pending_messages: dict[int, list[str]] = {}
user_message_timers: dict[int, asyncio.Task] = {}

# Файл для сохранения состояния постоянного молчания
SILENCE_STATE_FILE = "silence_state.json"

# --- Initialize API Clients ---
try:
    openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("Клиент OpenAI инициализирован.")
except Exception as e:
    logger.critical(f"Не удалось инициализировать клиент OpenAI: {e}", exc_info=True)
    sys.exit(1)

try:
    vk_session_api = vk_api.VkApi(token=VK_GROUP_TOKEN, api_version=VK_API_VERSION)
    logger.info("VK API сессия инициализирована (СИНХРОННО). Long Poll будет инициализирован в своем потоке.")
except vk_api.AuthError as e:
     logger.critical(f"Ошибка авторизации VK: {e}. Проверьте токен группы.", exc_info=True)
     sys.exit(1)
except Exception as e:
    logger.critical(f"Ошибка инициализации VK API: {e}", exc_info=True)
    sys.exit(1)

vector_collection: Optional[chromadb.api.models.Collection.Collection] = None # Используем Optional

def _get_active_db_subpath() -> Optional[str]: # Используем Optional
    try:
        active_db_info_filepath = os.path.join(VECTOR_DB_BASE_PATH, ACTIVE_DB_INFO_FILE)
        if os.path.exists(active_db_info_filepath):
            with open(active_db_info_filepath, "r", encoding="utf-8") as f:
                active_subdir = f.read().strip()
            if active_subdir:
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
            logger.info(f"Файл информации об активной БД '{ACTIVE_DB_INFO_FILE}' не найден.")
            return None
    except Exception as e:
        logger.error(f"Ошибка при чтении файла информации об активной БД: {e}", exc_info=True)
        return None

async def _initialize_active_vector_collection():
    global vector_collection
    active_subdir = _get_active_db_subpath()
    if active_subdir:
        active_db_full_path = os.path.join(VECTOR_DB_BASE_PATH, active_subdir)
        try:
            os.makedirs(VECTOR_DB_BASE_PATH, exist_ok=True)
            os.makedirs(active_db_full_path, exist_ok=True) 
            
            chroma_client_init = chromadb.PersistentClient(path=active_db_full_path)
            vector_collection = chroma_client_init.get_or_create_collection(
                name=VECTOR_DB_COLLECTION_NAME,
            )
            logger.info(f"Успешно подключено к ChromaDB: '{active_db_full_path}'. Коллекция: '{VECTOR_DB_COLLECTION_NAME}'.")
            if vector_collection: # Добавлена проверка
                logger.info(f"Документов в активной коллекции при старте: {vector_collection.count()}")
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB для пути '{active_db_full_path}': {e}. Поиск по базе знаний будет недоступен.", exc_info=True)
            vector_collection = None
    else:
        logger.warning("Не удалось определить активную директорию БД. База знаний будет недоступна.")
        vector_collection = None

def get_drive_service():
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

drive_service = get_drive_service()

# --- Helper Functions ---
def get_user_key(user_id: int) -> str:
    return str(user_id)

def is_non_working_hours() -> bool:
    now_local = datetime.now(TARGET_TZ)
    current_time_local = now_local.time()
    is_non_working = current_time_local >= WORK_END_TIME or current_time_local < WORK_START_TIME
    return is_non_working

async def send_vk_message(peer_id: int, message: str):
    if not message:
        logger.warning(f"Попытка отправить пустое сообщение в peer_id={peer_id}")
        return
    try:
        current_random_id = vk_api.utils.get_random_id()
        MY_PENDING_RANDOM_IDS.add(current_random_id)
        logger.debug(f"Добавлен random_id {current_random_id} в MY_PENDING_RANDOM_IDS для peer_id={peer_id}")

        await asyncio.to_thread(
            vk_session_api.method,
            'messages.send',
            {
                'peer_id': peer_id,
                'message': message,
                'random_id': current_random_id
            }
        )
    except vk_api.exceptions.ApiError as e:
        logger.error(f"Ошибка VK API при отправке сообщения в peer_id={peer_id}: {e}", exc_info=True)
        if 'current_random_id' in locals() and current_random_id in MY_PENDING_RANDOM_IDS:
            MY_PENDING_RANDOM_IDS.remove(current_random_id)
            logger.debug(f"Удален random_id {current_random_id} из MY_PENDING_RANDOM_IDS из-за ошибки отправки для peer_id={peer_id}")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при отправке сообщения в peer_id={peer_id}: {e}", exc_info=True)
        if 'current_random_id' in locals() and current_random_id in MY_PENDING_RANDOM_IDS:
            MY_PENDING_RANDOM_IDS.remove(current_random_id)
            logger.debug(f"Удален random_id {current_random_id} из MY_PENDING_RANDOM_IDS из-за ошибки отправки для peer_id={peer_id}")

async def set_typing_activity(peer_id: int):
     try:
        await asyncio.to_thread(
             vk_session_api.method,
             'messages.setActivity',
             {'type': 'typing', 'peer_id': peer_id}
         )
     except Exception as e:
         logger.warning(f"Не удалось установить статус 'typing' для peer_id={peer_id}: {e}")

# --- Silence Mode Management (Permanent Only) ---

async def save_silence_state_to_file():
    """Сохраняет текущее состояние постоянных режимов молчания в JSON-файл."""
    logger.debug("Сохранение состояния постоянных режимов молчания в файл...")
    data_to_save = {str(peer_id): True for peer_id in chat_silence_state if chat_silence_state[peer_id]}
    try:
        def _save():
            with open(SILENCE_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4)
        await asyncio.to_thread(_save)
        logger.info(f"Состояние постоянных режимов молчания сохранено в {SILENCE_STATE_FILE}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении состояния режимов молчания: {e}", exc_info=True)

async def load_silence_state_from_file():
    """Загружает состояние постоянных режимов молчания из JSON-файла при старте бота."""
    global chat_silence_state
    logger.info("Загрузка состояния постоянных режимов молчания из файла...")
    try:
        def _load():
            if not os.path.exists(SILENCE_STATE_FILE):
                logger.info(f"Файл {SILENCE_STATE_FILE} не найден. Пропускаем загрузку.")
                return None
            with open(SILENCE_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        
        loaded_data = await asyncio.to_thread(_load)

        if not loaded_data:
            return

        restored_count = 0
        for peer_id_str, should_be_silent in loaded_data.items():
            try:
                peer_id = int(peer_id_str)
                if should_be_silent:
                    chat_silence_state[peer_id] = True
                    logger.info(f"Восстановлен постоянный режим молчания для peer_id={peer_id}")
                    restored_count += 1
            except (ValueError, KeyError) as e: # Добавлен KeyError
                logger.error(f"Ошибка при обработке записи для peer_id_str='{peer_id_str}': {e}", exc_info=True)
        
        if restored_count > 0:
            logger.info(f"Успешно восстановлено {restored_count} состояний постоянного молчания.")
        else:
            logger.info("Активных состояний постоянного молчания для восстановления не найдено.")

    except FileNotFoundError:
        logger.info(f"Файл {SILENCE_STATE_FILE} не найден. Запуск с чистым состоянием молчания.")
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON из файла {SILENCE_STATE_FILE}. Возможно, файл поврежден.")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при загрузке состояния режимов молчания: {e}", exc_info=True)

async def silence_user(peer_id: int):
    """Активирует ПОСТОЯННЫЙ режим молчания для пользователя/чата (обычно из-за CRM)."""
    if chat_silence_state.get(peer_id):
        logger.info(f"Постоянный режим молчания для peer_id={peer_id} уже был активен.")
        return

    logger.info(f"Активация постоянного режима молчания для peer_id={peer_id}.")
    chat_silence_state[peer_id] = True
    await save_silence_state_to_file()

async def unsilence_user(peer_id: int):
    """Деактивирует ПОСТОЯННЫЙ режим молчания для пользователя/чата (командой 'speak')."""
    if peer_id in chat_silence_state:
        logger.info(f"Ручная деактивация (командой speak) режима молчания для peer_id={peer_id}.")
        chat_silence_state.pop(peer_id)
        await save_silence_state_to_file()
    else:
         logger.debug(f"Попытка снять молчание для peer_id={peer_id}, но бот и так был активен (постоянное молчание не было установлено).")

# --- Функции буферизации сообщений ---
async def schedule_buffered_processing(peer_id: int, original_user_id: int):
    log_prefix = f"schedule_buffered_processing(peer:{peer_id}, user:{original_user_id}):"
    current_task = asyncio.current_task()
    try:
        logger.debug(f"{log_prefix} Ожидание {MESSAGE_BUFFER_SECONDS} секунд...")
        await asyncio.sleep(MESSAGE_BUFFER_SECONDS)
        task_in_dict = user_message_timers.get(peer_id)
        if task_in_dict is not current_task:
            logger.info(f"{log_prefix} Таймер сработал, но он устарел. Обработка отменена.")
            return
        if peer_id in user_message_timers:
            del user_message_timers[peer_id]
        logger.debug(f"{log_prefix} Таймер сработал и удален. Вызов process_buffered_messages.")
        asyncio.create_task(process_buffered_messages(peer_id, original_user_id))
    except asyncio.CancelledError:
        logger.info(f"{log_prefix} Таймер отменен.")
    except Exception as e:
        logger.error(f"{log_prefix} Ошибка в задаче таймера: {e}", exc_info=True)
        if peer_id in user_message_timers and user_message_timers.get(peer_id) is current_task:
            del user_message_timers[peer_id]

async def process_buffered_messages(peer_id: int, original_user_id: int):
    log_prefix = f"process_buffered_messages(peer:{peer_id}, user:{original_user_id}):"
    logger.debug(f"{log_prefix} Начало обработки буферизованных сообщений.")
    async with user_processing_locks[peer_id]:
        logger.debug(f"{log_prefix} Блокировка для peer_id={peer_id} получена.")
        messages_to_process = pending_messages.pop(peer_id, [])
        if peer_id in user_message_timers:
            logger.warning(f"{log_prefix} Таймер для peer_id={peer_id} все еще существовал! Отменяем и удаляем.")
            timer_to_cancel = user_message_timers.pop(peer_id)
            if not timer_to_cancel.done():
                try:
                    timer_to_cancel.cancel()
                except Exception as e_inner_cancel:
                    logger.debug(f"{log_prefix} Ошибка при попытке отменить таймер: {e_inner_cancel}")
        if not messages_to_process:
            logger.info(f"{log_prefix} Нет сообщений в буфере для peer_id={peer_id}.")
            return
        combined_input = "\n".join(messages_to_process)
        num_messages = len(messages_to_process)
        logger.info(f'{log_prefix} Объединенный запрос для peer_id={peer_id} ({num_messages} сообщ.): "{combined_input[:200]}..."')
        try:
            await set_typing_activity(peer_id)
            response_text = await chat_with_assistant(original_user_id, combined_input)
            await send_vk_message(peer_id, response_text)
            logger.info(f"{log_prefix} Успешно обработан и отправлен ответ для peer_id={peer_id}.")
        except Exception as e:
            logger.error(f"{log_prefix} Ошибка при обработке или отправке ответа для peer_id={peer_id}: {e}", exc_info=True)
            try:
                await send_vk_message(peer_id, "Произошла внутренняя ошибка при обработке вашего запроса. Попробуйте позже.")
            except Exception as send_err_e:
                logger.error(f"{log_prefix} Не удалось отправить сообщение об ошибке peer_id={peer_id}: {send_err_e}")
        finally:
            logger.debug(f"{log_prefix} Блокировка для peer_id={peer_id} освобождена.")

# --- OpenAI Assistant Interaction ---
async def get_or_create_thread(user_id: int) -> Optional[str]: # Используем Optional
    user_key = get_user_key(user_id)
    if user_key in user_threads:
        thread_id = user_threads[user_key]
        try:
            await openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1)
            logger.info(f"Используем существующий тред {thread_id} для user_id={user_id}")
            return thread_id
        except openai.NotFoundError:
            logger.warning(f"Тред {thread_id} не найден в OpenAI для user_id={user_id}. Создаем новый.")
            del user_threads[user_key] # Удаляем невалидный ID
        except Exception as e:
            logger.error(f"Ошибка доступа к треду {thread_id} для user_id={user_id}: {e}. Создаем новый.")
            if user_key in user_threads: # Проверка перед удалением
                 del user_threads[user_key]
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
    thread_id = await get_or_create_thread(user_id)
    if not thread_id:
        return "Произошла внутренняя ошибка (не удалось создать тред)."
    try:
        context = ""
        if vector_collection:
             context = await get_relevant_context(message_text, k=RELEVANT_CONTEXT_COUNT)
             await log_context(user_id, message_text, context)
        full_prompt = message_text
        if context:
            full_prompt = f"Используй следующую информацию из базы знаний для ответа:\n--- НАЧАЛО КОНТЕКСТА ---\n{context}\n--- КОНЕЦ КОНТЕКСТА ---\n\nВопрос пользователя: {message_text}"
        else:
            logger.info(f"Контекст для запроса user_id={user_id} не найден или база знаний отключена.")
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
        await openai_client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=full_prompt
        )
        logger.info(f"Сообщение добавлено в тред {thread_id} для user_id={user_id}")
        run = await openai_client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=ASSISTANT_ID
        )
        logger.info(f"Запущен новый run {run.id} для треда {thread_id}")
        start_time = time_module.time()
        while time_module.time() - start_time < OPENAI_RUN_TIMEOUT_SECONDS:
            await asyncio.sleep(1)
            run_status = await openai_client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
            )
            if run_status.status == 'completed':
                logger.info(f"Run {run.id} успешно завершен.")
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                error_message = f"Run {run.id} завершился со статусом '{run_status.status}'."
                last_error = getattr(run_status, 'last_error', None)
                if last_error: error_message += f" Ошибка: {last_error.message} (Код: {last_error.code})"
                logger.error(error_message)
                return "Произошла внутренняя ошибка при обработке вашего запроса (статус OpenAI)."
            elif run_status.status == 'requires_action':
                 logger.warning(f"Run {run.id} требует действия (Function Calling?), что не поддерживается.")
                 await openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                 return "Произошла внутренняя ошибка при обработке вашего запроса (OpenAI requires_action)."
        else:
            logger.warning(f"Превышено время ожидания ({OPENAI_RUN_TIMEOUT_SECONDS}s) ответа от OpenAI для run {run.id}, тред {thread_id}")
            try:
                await openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                logger.info(f"Отменен run {run.id} из-за таймаута.")
            except Exception as cancel_error:
                logger.warning(f"Не удалось отменить run {run.id} после таймаута: {cancel_error}")
            return "Произошла внутренняя ошибка при обработке вашего запроса (таймаут OpenAI)."
        messages_response = await openai_client.beta.threads.messages.list(
            thread_id=thread_id, order="desc", limit=5
        )
        assistant_response = None
        for msg in messages_response.data:
            if msg.role == "assistant" and msg.run_id == run.id:
                if msg.content and msg.content[0].type == 'text':
                    assistant_response = msg.content[0].text.value
                    logger.info(f"Получен ответ от ассистента для user_id={user_id}: {assistant_response[:100]}...")
                    break
        if assistant_response:
            await log_context(user_id, message_text, context, assistant_response)
            return assistant_response
        else:
            logger.warning(f"Не найдено текстового ответа от ассистента в треде {thread_id} после run {run.id}. Ответы: {messages_response.data}")
            for msg in messages_response.data:
                 if msg.role == "assistant":
                      if msg.content and msg.content[0].type == 'text':
                           logger.warning(f"Найден ответ ассистента, но от другого run ({msg.run_id}) - используем его.")
                           return msg.content[0].text.value
            return "Произошла внутренняя ошибка (ответ ассистента не найден)."
    except openai.APIError as e:
         logger.error(f"OpenAI API ошибка для user_id={user_id}: {e}", exc_info=True)
         return "Произошла внутренняя ошибка (API OpenAI)."
    except Exception as e:
        logger.error(f"Непредвиденная ошибка в chat_with_assistant для user_id={user_id}: {e}", exc_info=True)
        return "Произошла внутренняя ошибка при обработке вашего запроса."

# --- Vector Store Management (ChromaDB) ---
async def get_relevant_context(query: str, k: int) -> str:
    if not vector_collection:
        logger.warning("Запрос контекста, но ChromaDB не инициализирована.")
        return ""
    try:
        try:
            query_embedding_response = await openai_client.embeddings.create(
                 input=[query],
                 model=EMBEDDING_MODEL,
                 dimensions=EMBEDDING_DIMENSIONS if EMBEDDING_DIMENSIONS else None
            )
            query_embedding = query_embedding_response.data[0].embedding
            logger.debug(f"Эмбеддинг для запроса '{query[:50]}...' создан.")
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга запроса: {e}", exc_info=True)
            return ""
        try:
            results = await asyncio.to_thread(
                vector_collection.query,
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            logger.debug(f"Поиск в ChromaDB для '{query[:50]}...' выполнен.")
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска в ChromaDB: {e}", exc_info=True)
            return ""
        if not results or not results.get("ids") or not results["ids"][0]:
            logger.info(f"Релевантных документов не найдено для запроса: '{query[:50]}...'")
            return ""
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        context_pieces = []
        logger.info(f"Найдено {len(documents)} док-в для '{query[:50]}...'. Топ {k}:")
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            source = meta.get('source', 'Неизвестный источник')
            logger.info(f"  #{i+1}: Источник='{source}', Дистанция={dist:.4f}, Контент='{doc[:100]}...'")
            context_piece = f"Из документа '{source}':\n{doc}"
            context_pieces.append(context_piece)
        if not context_pieces:
             logger.info(f"Не найдено подходящих фрагментов контекста для '{query[:50]}...'.")
             return ""
        full_context = "\n\n---\n\n".join(context_pieces)
        logger.info(f"Сформирован контекст размером {len(full_context)} символов из {len(context_pieces)} фрагментов.")
        return full_context
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при получении контекста: {e}", exc_info=True)
        return ""

async def update_vector_store():
    logger.info("--- Запуск обновления базы знаний ---")
    previous_active_subpath = _get_active_db_subpath()
    os.makedirs(VECTOR_DB_BASE_PATH, exist_ok=True)
    if not drive_service:
        logger.error("Обновление БЗ невозможно: сервис Google Drive не инициализирован.")
        return {"success": False, "error": "Сервис Google Drive не инициализирован", "added_chunks": 0, "total_chunks": 0}
    timestamp_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_new"
    new_db_path = os.path.join(VECTOR_DB_BASE_PATH, timestamp_dir_name)
    logger.info(f"Создание новой временной директории для БД: {new_db_path}")
    try:
        os.makedirs(new_db_path, exist_ok=True)
    except Exception as e_mkdir:
        logger.error(f"Не удалось создать временную директорию '{new_db_path}': {e_mkdir}.", exc_info=True)
        return {"success": False, "error": f"Failed to create temp dir: {e_mkdir}", "added_chunks": 0, "total_chunks": 0}
    temp_vector_collection = None
    try:
        temp_chroma_client = chromadb.PersistentClient(path=new_db_path)
        temp_vector_collection = temp_chroma_client.get_or_create_collection(name=VECTOR_DB_COLLECTION_NAME)
        logger.info(f"Временная коллекция '{VECTOR_DB_COLLECTION_NAME}' создана/получена в '{new_db_path}'.")
        logger.info("Получение данных из Google Drive...")
        documents_data = await asyncio.to_thread(read_data_from_drive)
        if not documents_data:
            logger.warning("Не найдено документов в Google Drive. Обновление прервано.")
            if os.path.exists(new_db_path): shutil.rmtree(new_db_path)
            return {"success": False, "error": "No documents in Google Drive", "added_chunks": 0, "total_chunks": 0}
        logger.info(f"Получено {len(documents_data)} документов из Google Drive.")
        all_texts, all_metadatas = [], []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
        MD_SECTION_MAX_LEN = 2000
        for doc_info in documents_data:
            doc_name, doc_content = doc_info['name'], doc_info['content']
            if not doc_content or not doc_content.strip():
                logger.warning(f"Документ '{doc_name}' пуст. Пропускаем.")
                continue
            enhanced_doc_content = f"Документ: {doc_name}\n\n{doc_content}"
            chunk_idx = 0
            is_md = doc_name.lower().endswith(('.md', '.markdown'))
            try:
                if is_md:
                    md_splits = markdown_splitter.split_text(enhanced_doc_content)
                    for md_split in md_splits:
                        headers_meta = {k: v for k, v in md_split.metadata.items() if k.startswith('h')}
                        if len(md_split.page_content) > MD_SECTION_MAX_LEN:
                            sub_chunks = text_splitter.split_text(md_split.page_content)
                            for sub_chunk_text in sub_chunks:
                                all_texts.append(sub_chunk_text)
                                all_metadatas.append({"source": doc_name, **headers_meta, "type": "md_split", "chunk": chunk_idx})
                                chunk_idx += 1
                        else:
                            all_texts.append(md_split.page_content)
                            all_metadatas.append({"source": doc_name, **headers_meta, "type": "md", "chunk": chunk_idx})
                            chunk_idx += 1
                else:
                    chunks = text_splitter.split_text(enhanced_doc_content)
                    for chunk_text in chunks:
                        all_texts.append(chunk_text)
                        all_metadatas.append({"source": doc_name, "type": "text", "chunk": chunk_idx})
                        chunk_idx += 1
                logger.info(f"Документ '{doc_name}' разбит на {chunk_idx} чанков.")
            except Exception as e_split:
                logger.error(f"Ошибка при разбиении '{doc_name}': {e_split}", exc_info=True)
                # Fallback to simple text splitting if markdown fails
                if is_md:
                    try:
                        chunks = text_splitter.split_text(enhanced_doc_content)
                        chunk_idx = 0
                        for chunk_text in chunks:
                            all_texts.append(chunk_text)
                            all_metadatas.append({"source": doc_name, "type": "text_fallback", "chunk": chunk_idx})
                            chunk_idx += 1
                        logger.info(f"Документ '{doc_name}' (fallback) разбит на {chunk_idx} чанков.")
                    except Exception as e_fallback:
                         logger.error(f"Ошибка fallback-разбиения '{doc_name}': {e_fallback}", exc_info=True)
                continue
        if not all_texts:
            logger.warning("Нет текстовых данных для добавления в базу. Обновление прервано.")
            if os.path.exists(new_db_path): shutil.rmtree(new_db_path)
            return {"success": False, "error": "No text data to add", "added_chunks": 0, "total_chunks": 0}
        logger.info(f"Добавление {len(all_texts)} чанков во временную коллекцию...")
        try:
            all_ids = [f"{meta['source']}_{meta['chunk']}_{random.randint(1000,9999)}" for meta in all_metadatas] # Ensure unique IDs
            logger.info(f"Создание эмбеддингов для {len(all_texts)} чанков...")
            embeddings_response = await openai_client.embeddings.create(
                input=all_texts, model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS if EMBEDDING_DIMENSIONS else None
            )
            all_embeddings = [item.embedding for item in embeddings_response.data]
            await asyncio.to_thread(
               temp_vector_collection.add, # type: ignore
               ids=all_ids, embeddings=all_embeddings, metadatas=all_metadatas, documents=all_texts
            )
            final_added, final_total = len(all_ids), temp_vector_collection.count() # type: ignore
            logger.info(f"Успешно добавлено {final_added} чанков. Всего: {final_total}.")
            active_db_info_filepath = os.path.join(VECTOR_DB_BASE_PATH, ACTIVE_DB_INFO_FILE)
            with open(active_db_info_filepath, "w", encoding="utf-8") as f: f.write(timestamp_dir_name)
            logger.info(f"Путь к новой активной базе '{timestamp_dir_name}' сохранен.")
            await _initialize_active_vector_collection()
            if not vector_collection:
                 return {"success": False, "error": "Failed to reload global vector_collection", "added_chunks": final_added, "total_chunks": final_total}
            if previous_active_subpath and previous_active_subpath != timestamp_dir_name:
                prev_path = os.path.join(VECTOR_DB_BASE_PATH, previous_active_subpath)
                if os.path.exists(prev_path):
                    shutil.rmtree(prev_path)
                    logger.info(f"Удалена предыдущая БД: '{prev_path}'")
            logger.info("--- Обновление базы знаний успешно завершено ---")
            return {"success": True, "added_chunks": final_added, "total_chunks": final_total, "new_active_path": timestamp_dir_name}
        except openai.APIError as e_openai:
             logger.error(f"OpenAI API ошибка при эмбеддингах: {e_openai}", exc_info=True)
             if os.path.exists(new_db_path): shutil.rmtree(new_db_path)
             return {"success": False, "error": f"OpenAI API error: {e_openai}", "added_chunks": 0, "total_chunks": 0}
        except Exception as e_add:
            logger.error(f"Ошибка при добавлении в ChromaDB: {e_add}", exc_info=True)
            if os.path.exists(new_db_path): shutil.rmtree(new_db_path)
            return {"success": False, "error": f"ChromaDB add error: {e_add}", "added_chunks": 0, "total_chunks": 0}
    except Exception as e_main_update:
        logger.error(f"Критическая ошибка во время обновления БЗ: {e_main_update}", exc_info=True)
        if os.path.exists(new_db_path): shutil.rmtree(new_db_path)
        return {"success": False, "error": f"Critical update error: {e_main_update}", "added_chunks": 0, "total_chunks": 0}

# --- Google Drive Reading ---
def read_data_from_drive() -> list[dict]:
    if not drive_service:
        logger.error("Чтение из Google Drive невозможно: сервис не инициализирован.")
        return []
    result_docs = []
    try:
        files_response = drive_service.files().list(
            q=f"'{FOLDER_ID}' in parents and trashed=false",
            fields="files(id, name, mimeType)", pageSize=1000
        ).execute()
        files = files_response.get('files', [])
        logger.info(f"Найдено {len(files)} файлов в папке Google Drive.")
        downloader_map = {
            'application/vnd.google-apps.document': download_google_doc,
            'application/pdf': download_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': download_docx,
            'text/plain': download_text,
            'text/markdown': download_text, # Treat .md as plain text for download
        }
        for file_item in files: # Renamed to file_item to avoid conflict
            file_id, mime_type, file_name = file_item['id'], file_item['mimeType'], file_item['name']
            if mime_type in downloader_map:
                logger.info(f"Обработка файла: '{file_name}' (ID: {file_id}, Type: {mime_type})")
                try:
                    content = downloader_map[mime_type](drive_service, file_id)
                    if content and content.strip():
                        result_docs.append({'name': file_name, 'content': content})
                        logger.info(f"Успешно прочитан файл: '{file_name}' ({len(content)} симв)")
                    else:
                        logger.warning(f"Файл '{file_name}' пуст или не удалось извлечь контент.")
                except Exception as e_read_file:
                    logger.error(f"Ошибка чтения файла '{file_name}': {e_read_file}", exc_info=True)
            else:
                logger.debug(f"Файл '{file_name}' имеет неподдерживаемый тип ({mime_type}).")
    except Exception as e:
        logger.error(f"Критическая ошибка при чтении из Google Drive: {e}", exc_info=True)
        return []
    logger.info(f"Чтение из Google Drive завершено. Прочитано {len(result_docs)} документов.")
    return result_docs

def download_google_doc(service, file_id):
    request = service.files().export_media(fileId=file_id, mimeType='text/plain')
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        logger.debug(f"Google Doc Downloader: file_id={file_id}, status={status.progress() if status else 'N/A'}")
    return fh.getvalue().decode('utf-8', errors='ignore')

def download_pdf(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        logger.debug(f"PDF Downloader: file_id={file_id}, status={status.progress() if status else 'N/A'}")
    fh.seek(0)
    try:
        pdf_reader = PyPDF2.PdfReader(fh)
        return "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
    except Exception as e:
         logger.error(f"Ошибка обработки PDF (ID: {file_id}): {e}", exc_info=True)
         return ""

def download_docx(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        logger.debug(f"DOCX Downloader: file_id={file_id}, status={status.progress() if status else 'N/A'}")
    fh.seek(0)
    try:
        doc = docx.Document(fh)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
    except Exception as e:
         logger.error(f"Ошибка обработки DOCX (ID: {file_id}): {e}", exc_info=True)
         return ""

def download_text(service, file_id): # Handles both text/plain and text/markdown
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    MediaIoBaseDownload(fh, request).next_chunk() # Simplified
    try:
        return fh.getvalue().decode('utf-8')
    except UnicodeDecodeError:
         logger.warning(f"Не удалось декодировать {file_id} как UTF-8, пробуем cp1251.")
         try: return fh.getvalue().decode('cp1251', errors='ignore')
         except Exception as e:
              logger.error(f"Не удалось декодировать {file_id}: {e}")
              return ""

# --- History and Context Management ---
async def log_context(user_id, message_text, context, response_text=None):
    try:
        ts = datetime.now()
        log_filename = os.path.join(LOGS_DIR, f"context_{user_id}_{ts.strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {ts.isoformat()}\nUser ID: {user_id}\n"
                    f"--- User Query ---\n{message_text}\n"
                    f"--- Retrieved Context ---\n{context or 'Контекст не найден.'}\n")
            if response_text: f.write(f"--- Assistant Response ---\n{response_text}\n")
    except Exception as e:
        logger.error(f"Ошибка при логировании контекста для user_id={user_id}: {e}", exc_info=True)

async def cleanup_old_context_logs():
    logger.info("Запуск очистки старых логов контекста...")
    count = 0
    try:
        cutoff = time_module.time() - LOG_RETENTION_SECONDS
        for filename in glob.glob(os.path.join(LOGS_DIR, "context_*.log")):
            try:
                if os.path.getmtime(filename) < cutoff:
                    os.remove(filename)
                    count += 1
            except Exception: continue
        logger.info(f"Очистка логов: удалено {count} устаревших файлов." if count else "Устаревших логов не найдено.")
    except Exception as e:
        logger.error(f"Критическая ошибка при очистке логов контекста: {e}", exc_info=True)

# --- Background Cleanup Task ---
last_auto_update_date: Optional[datetime.date] = None

async def background_cleanup_task():
    global last_auto_update_date
    while True:
        await asyncio.sleep(3600) # Каждый час
        logger.info("Запуск периодической фоновой задачи...")
        try:
            now_local = datetime.now(TARGET_TZ)
            if now_local.hour == 4 and (last_auto_update_date is None or last_auto_update_date < now_local.date()):
                logger.info(f"Время для ежедневного обновления БЗ ({now_local.hour}:00). Запускаем...")
                await run_update_and_notify_admin(ADMIN_USER_ID)
                last_auto_update_date = now_local.date()
        except Exception as e_auto_update:
            logger.error(f"Ошибка в логике ежедневного обновления БЗ: {e_auto_update}", exc_info=True)
        await cleanup_old_context_logs()
        logger.info("Периодическая фоновая задача завершила цикл.")

# --- Main Event Handler ---
async def handle_new_message(event: VkBotEvent):
    global user_threads
    try:
        if event.from_user:
            user_id = event.obj.message['from_id']
            peer_id = event.obj.message['peer_id']
            message_text = event.obj.message['text'].strip()
            if not message_text:
                 logger.info(f"Пустое сообщение от user_id={user_id}. Игнорируем.")
                 return

            if message_text.lower() == "/update" and user_id == ADMIN_USER_ID:
                logger.info(f"Администратор {user_id} инициировал обновление БЗ.")
                await send_vk_message(peer_id, "🔄 Запускаю обновление базы знаний...")
                asyncio.create_task(run_update_and_notify_admin(peer_id)) 
                return
            
            if message_text.lower() == "/reset":
                user_key = get_user_key(user_id)
                log_prefix = f"handle_new_message(reset for peer:{peer_id}, user:{user_id}):"
                logging.info(f"{log_prefix} Получена команда сброса диалога.")
                if peer_id in pending_messages: del pending_messages[peer_id]
                if peer_id in user_message_timers:
                    old_timer = user_message_timers.pop(peer_id)
                    if not old_timer.done(): old_timer.cancel()
                thread_id_to_forget = user_threads.pop(user_key, None)
                if thread_id_to_forget: logging.info(f"{log_prefix} Тред {thread_id_to_forget} удален из памяти.")
                await send_vk_message(peer_id, "🔄 Диалог сброшен.")
                return
            
            if message_text.lower() == "/reset_all" and user_id == ADMIN_USER_ID:
                log_prefix = f"handle_new_message(reset_all from user:{user_id}):"
                logging.info(f"{log_prefix} Получена команда сброса ВСЕХ диалогов.")
                active_timer_count = sum(1 for task in user_message_timers.values() if not task.done())
                for task in user_message_timers.values():
                    if not task.done(): task.cancel()
                user_message_timers.clear()
                pending_count = len(pending_messages)
                pending_messages.clear()
                threads_count = len(user_threads)
                user_threads.clear()
                await send_vk_message(peer_id, f"🔄 СБРОС ВСЕХ ДИАЛОГОВ ВЫПОЛНЕН.\n- Таймеров: {active_timer_count}\n- Буферов: {pending_count}\n- Тредов: {threads_count}")
                return

            is_manager = user_id in MANAGER_USER_IDS or user_id == ADMIN_USER_ID
            if is_manager:
                command = message_text.lower()
                if command == CMD_SPEAK.lower(): # Только команда speak
                    await unsilence_user(peer_id) # Снимаем постоянное молчание
                    await send_vk_message(peer_id, "🤖 Режим молчания снят. Бот снова активен.")
                    return

            if chat_silence_state.get(peer_id, False): # Проверка на постоянное молчание
                logger.info(f"Бот в режиме молчания для peer_id={peer_id} (CRM). Сообщение от user_id={user_id} игнорируется.")
                return

            now_dt = datetime.now()
            last_time = user_last_message_time.get(user_id)
            if last_time and now_dt - last_time < timedelta(seconds=MESSAGE_COOLDOWN_SECONDS):
                logger.warning(f"Кулдаун для user_id={user_id}. Игнорируем.")
                return
            user_last_message_time[user_id] = now_dt

            logger.info(f"Получено сообщение от user_id={user_id} (peer_id={peer_id}): '{message_text[:100]}...'")
            pending_messages.setdefault(peer_id, []).append(message_text)
            logger.debug(f"Сообщение от peer_id={peer_id} добавлено в буфер: {pending_messages[peer_id]}")
            if peer_id in user_message_timers:
                old_timer = user_message_timers.pop(peer_id)
                if not old_timer.done():
                    try: old_timer.cancel()
                    except Exception as e_cancel: logger.warning(f"Не удалось отменить таймер: {e_cancel}")
            logger.debug(f"Запуск таймера буферизации для peer_id={peer_id} ({MESSAGE_BUFFER_SECONDS} сек).")
            new_timer_task = asyncio.create_task(schedule_buffered_processing(peer_id, user_id))
            user_message_timers[peer_id] = new_timer_task
        
        elif event.from_chat:
            chat_id = event.chat_id # type: ignore
            user_id = event.obj.message['from_id']
            message_text = event.obj.message['text'].strip()
            logger.debug(f"Сообщение в чате {chat_id} от {user_id}: {message_text[:50]}")
            # Логика для групповых чатов не реализована для ответа бота
            pass
        else:
            logger.debug(f"Получено событие Long Poll типа {event.type}, не обрабатывается.")
    except Exception as e:
        logger.error(f"Критическая ошибка в handle_new_message: {e}", exc_info=True)

# --- Main Application Logic ---
async def run_update_and_notify_admin(notification_peer_id: int):
    logger.info(f"run_update_and_notify_admin: Запуск обновления БЗ для peer_id={notification_peer_id}")
    update_result = await update_vector_store()
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    admin_message = f"🔔 Отчет об обновлении БЗ ({current_time_str}):\n"
    if update_result.get("success"):
        admin_message += (f"✅ Успешно!\n➕ Добавлено: {update_result.get('added_chunks', 'N/A')}\n"
                          f"📊 Всего: {update_result.get('total_chunks', 'N/A')}\n")
        if update_result.get("new_active_path"): admin_message += f"📁 Путь: {update_result['new_active_path']}"
    else:
        admin_message += f"❌ Ошибка: {update_result.get('error', 'N/A')}\nБаза могла не измениться."
    logger.info(f"Результат обновления БЗ: {admin_message}")
    try:
        await send_vk_message(notification_peer_id, admin_message)
        if ADMIN_USER_ID != 0 and notification_peer_id != ADMIN_USER_ID and ADMIN_USER_ID > 0:
            await send_vk_message(ADMIN_USER_ID, "[Авто] " + admin_message)
    except Exception as e_notify:
        logger.error(f"Не удалось отправить уведомление админу: {e_notify}", exc_info=True)

async def main():
    logger.info("--- Запуск VK бота ---")
    await load_silence_state_from_file() # Загружаем состояние молчания
    await _initialize_active_vector_collection()
    logger.info("Запуск фонового обновления БЗ при старте...")
    asyncio.create_task(run_update_and_notify_admin(ADMIN_USER_ID))
    cleanup_task = asyncio.create_task(background_cleanup_task())
    logger.info("Фоновая задача очистки запущена.")
    logger.warning("Используется СИНХРОННЫЙ VkBotLongPoll. Это БЛОКИРУЕТ асинхронный цикл.")
    listen_task = None
    try:
        loop = asyncio.get_running_loop()
        listen_task = asyncio.create_task(asyncio.to_thread(run_longpoll_sync, loop), name="VKLongPollListener")
        if listen_task: await listen_task
    except vk_api.exceptions.ApiError as e:
        logger.critical(f"Критическая ошибка VK API в Long Poll: {e}", exc_info=True)
    except Exception as e:
         logger.critical(f"Критическая ошибка в главном цикле: {e}", exc_info=True)
    finally:
        logger.info("Завершение работы фоновых задач...")
        if 'cleanup_task' in locals() and cleanup_task: cleanup_task.cancel() # type: ignore
        if listen_task and not listen_task.done():
             listen_task.cancel()
             logger.warning("Запрошена отмена задачи Long Poll.")
        await asyncio.gather(
            cleanup_task if 'cleanup_task' in locals() and cleanup_task else asyncio.sleep(0), # type: ignore
            listen_task if listen_task else asyncio.sleep(0),
            return_exceptions=True
        )
        logger.info("--- Бот остановлен ---")

def run_longpoll_sync(async_loop: asyncio.AbstractEventLoop):
    logger.info("Запуск синхронного Long Poll в отдельном потоке...")
    MAX_RECONNECT_ATTEMPTS, RECONNECT_DELAY_SECONDS = 5, 30
    current_attempts = 0
    global vk_session_api, VK_GROUP_ID # Убедимся, что они доступны
    
    while True:
        try:
            if not vk_session_api:
                logger.error("[Thread LongPoll] vk_session_api не инициализирована.")
                time_module.sleep(RECONNECT_DELAY_SECONDS * 5)
                continue

            logger.info(f"[Thread LongPoll] Инициализация VkBotLongPoll (попытка {current_attempts + 1}).")
            current_longpoll = VkBotLongPoll(vk_session_api, VK_GROUP_ID) # type: ignore
            logger.info("[Thread LongPoll] VkBotLongPoll инициализирован.")
            current_attempts = 0
            logger.info("[Thread LongPoll] Начало прослушивания событий...")
            for event in current_longpoll.listen():
                if event.type == VkBotEventType.MESSAGE_NEW:
                    asyncio.run_coroutine_threadsafe(handle_new_message(event), async_loop)
                elif event.type == VkBotEventType.MESSAGE_REPLY:
                    logger.debug(f"Получено MESSAGE_REPLY: {event.obj}")
                    try:
                        # Проверяем, что сообщение от нашей группы (исходящее)
                        # и from_id это ID нашей группы (отрицательный)
                        is_outgoing_from_group = (event.obj.get('out') == 1 and 
                                                  event.obj.get('from_id') == -int(VK_GROUP_ID)) # type: ignore
                        
                        if is_outgoing_from_group:
                            event_random_id = event.obj.get('random_id')
                            peer_id = event.obj.get('peer_id')

                            if event_random_id is not None and event_random_id in MY_PENDING_RANDOM_IDS:
                                MY_PENDING_RANDOM_IDS.remove(event_random_id)
                                logger.debug(f"MESSAGE_REPLY от бота (random_id: {event_random_id}) для peer_id={peer_id}. Удален из MY_PENDING_RANDOM_IDS.")
                            else:
                                crm_message_text = event.obj.get('text', '') 
                                logger.info(f"MESSAGE_REPLY от CRM/оператора (текст: '{crm_message_text[:50]}...', random_id: {event_random_id}) для peer_id={peer_id}. Активируем ПОСТОЯННЫЙ режим молчания.")
                                if peer_id:
                                    # Активируем постоянное молчание
                                    asyncio.run_coroutine_threadsafe(silence_user(peer_id), async_loop)
                                else:
                                    logger.warning(f"Не удалось определить peer_id из MESSAGE_REPLY для CRM: {event.obj}")
                        else:
                             logger.debug(f"Пропускаем MESSAGE_REPLY (не от нашей группы или не исходящее): {event.obj}")
                    except Exception as e_reply_proc:
                        logger.error(f"Ошибка при обработке MESSAGE_REPLY: {e_reply_proc}", exc_info=True)
                        logger.debug(f"Ошибочный MESSAGE_REPLY: {event.obj}")
                else:
                    logger.debug(f"Пропускаем событие типа {event.type}")
            logger.warning("[Thread LongPoll] Цикл listen() завершился. Перезапуск...")
            current_attempts = 0
            time_module.sleep(RECONNECT_DELAY_SECONDS)
        except (requests.exceptions.RequestException, vk_api.exceptions.VkApiError) as e_net: # Более общие сетевые ошибки
            logger.error(f"[Thread LongPoll] Ошибка сети/VK API: {e_net}", exc_info=True)
            current_attempts += 1
            if MAX_RECONNECT_ATTEMPTS > 0 and current_attempts >= MAX_RECONNECT_ATTEMPTS:
                logger.critical(f"[Thread LongPoll] Превышено макс. попыток переподключения. Остановка.")
                asyncio.run_coroutine_threadsafe(send_vk_message(ADMIN_USER_ID, "Критическая ошибка: VK Long Poll остановлен."), async_loop) # type: ignore
                break
            logger.info(f"[Thread LongPoll] Пауза {RECONNECT_DELAY_SECONDS}с перед попыткой {current_attempts + 1}...")
            time_module.sleep(RECONNECT_DELAY_SECONDS)
        except Exception as e_fatal:
            logger.critical(f"[Thread LongPoll] Непредвиденная критическая ошибка: {e_fatal}", exc_info=True)
            current_attempts += 1 # Считаем попытку
            if MAX_RECONNECT_ATTEMPTS > 0 and current_attempts >= MAX_RECONNECT_ATTEMPTS:
                 logger.critical(f"[Thread LongPoll] Превышено макс. попыток после непредвиденной ошибки. Остановка.")
                 asyncio.run_coroutine_threadsafe(send_vk_message(ADMIN_USER_ID, "Критическая ошибка: VK Long Poll остановлен (непредвиденная ошибка)."), async_loop) # type: ignore
                 break
            logger.info(f"[Thread LongPoll] Пауза {RECONNECT_DELAY_SECONDS * 2}с перед попыткой {current_attempts + 1}...")
            time_module.sleep(RECONNECT_DELAY_SECONDS * 2)
    logger.info("[Thread LongPoll] Поток Long Poll завершен.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получен сигнал KeyboardInterrupt. Завершение работы...")
    except Exception as e:
         logger.critical(f"Критическая ошибка при запуске: {e}", exc_info=True)