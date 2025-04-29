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
from datetime import datetime, timedelta, time as dt_time
import pytz # Добавим эту строку в начало файла, где импорты

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
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
VECTOR_DB_PATH = "./local_vector_db"
VECTOR_DB_COLLECTION = "documents"
RELEVANT_CONTEXT_COUNT = 5 # Количество релевантных чанков для контекста

# Bot Behavior Settings
MESSAGE_LIFETIME_DAYS = 100 # Срок хранения истории сообщений (в памяти)
MESSAGE_COOLDOWN_SECONDS = 3 # Минимальный интервал между сообщениями от пользователя
MANAGER_ACTIVE_TIMEOUT_SECONDS = 86400 # 24 часа - время молчания бота после команды менеджера
LOG_RETENTION_SECONDS = 86400 # 24 часа - время хранения логов контекста
OPENAI_RUN_TIMEOUT_SECONDS = 90 # Таймаут ожидания ответа от OpenAI Assistant

# Time Settings
CHELYABINSK_TZ_STR = "Asia/Yekaterinburg" # Часовой пояс Челябинска
# import time  # <--- Удалите или закомментируйте эту строку

try:
    CHELYABINSK_TZ = pytz.timezone(CHELYABINSK_TZ_STR)
except pytz.UnknownTimeZoneError:
    logger.error(f"Неизвестный часовой пояс: {CHELYABINSK_TZ_STR}. Проверьте написание.")
    # Можно установить UTC как запасной вариант или выйти
    CHELYABINSK_TZ = pytz.utc
# Время начала и конца РАБОЧЕГО дня
WORK_START_TIME = dt_time(9, 45, tzinfo=CHELYABINSK_TZ)
WORK_END_TIME = dt_time(19, 15, tzinfo=CHELYABINSK_TZ)

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
user_messages: dict[str, list[dict]] = {} # {user_key: [{'role': '...', 'content': '...', 'timestamp': ...}]}
user_processing_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock) # {peer_id: Lock}
user_last_message_time: dict[int, datetime] = {} # {user_id: timestamp}
chat_silence_state: dict[int, bool] = {} # {peer_id: True if silent}
chat_silence_timers: dict[int, asyncio.Task] = {} # {peer_id: silence timer task}

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
    longpoll = VkBotLongPoll(vk_session_api, VK_GROUP_ID)
    logger.info("VK API сессия и Long Poll инициализированы (СИНХРОННО).")
except vk_api.AuthError as e:
     logger.critical(f"Ошибка авторизации VK: {e}. Проверьте токен группы.", exc_info=True)
     sys.exit(1)
except Exception as e:
    logger.critical(f"Ошибка инициализации VK API: {e}", exc_info=True)
    sys.exit(1)

# ChromaDB Client
try:
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    # Пробуем получить/создать коллекцию при старте
    vector_collection = chroma_client.get_or_create_collection(
        name=VECTOR_DB_COLLECTION,
        # Указание embedding function не обязательно, если создаем эмбеддинги сами
        # metadata={"hnsw:space": "cosine"} # Пример настройки индекса
    )
    logger.info(f"Клиент ChromaDB подключен к '{VECTOR_DB_PATH}'. Коллекция: '{VECTOR_DB_COLLECTION}'.")
    logger.info(f"Документов в коллекции при старте: {vector_collection.count()}")
except Exception as e:
    logger.error(f"Ошибка инициализации ChromaDB: {e}. Поиск по базе знаний будет недоступен.", exc_info=True)
    vector_collection = None # Поиск будет отключен

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
    """Проверяет, является ли текущее время нерабочим в Челябинске."""
    # Используем datetime.now() вместо time.time() для работы с часовыми поясами
    now_local = datetime.now(CHELYABINSK_TZ) # datetime здесь - это импортированный класс datetime.datetime
    current_time_local = now_local.time() # now_local.time() возвращает объект datetime.time
    # Нерабочее время: ПОСЛЕ 19:15 ИЛИ ДО 9:45
    # Сравниваем время без информации о часовом поясе, т.к. now_local уже в нужном поясе
    # Используем объекты dt_time, созданные выше
    is_non_working = current_time_local >= WORK_END_TIME.replace(tzinfo=None) or current_time_local < WORK_START_TIME.replace(tzinfo=None)
    # Логируем для отладки (можно будет убрать потом)
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
            if user_key in user_messages: del user_messages[user_key]

    # Создаем новый тред
    try:
        logger.info(f"Создаем новый тред для user_id={user_id}...")
        thread = await openai_client.beta.threads.create()
        thread_id = thread.id
        user_threads[user_key] = thread_id
        user_messages[user_key] = [] # Инициализируем историю
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
        return "Извините, произошла ошибка при инициализации диалога. Пожалуйста, попробуйте позже."

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
                return f"Извините, произошла ошибка при обработке вашего запроса (статус: {run_status.status}). Попробуйте еще раз."
            elif run_status.status == 'requires_action':
                 logger.warning(f"Run {run.id} требует действия (Function Calling?), что не поддерживается в текущей конфигурации.")
                 # TODO: Обработать Function Calling, если он настроен у ассистента
                 # Пока просто прерываем как ошибку
                 await openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                 return "Извините, ассистент запросил действие, которое пока не поддерживается."

        else: # Сработал таймаут цикла while
            logger.warning(f"Превышено время ожидания ({OPENAI_RUN_TIMEOUT_SECONDS}s) ответа от ассистента для run {run.id}, тред {thread_id}")
            try:
                await openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                logger.info(f"Отменен run {run.id} из-за таймаута.")
            except Exception as cancel_error:
                logger.warning(f"Не удалось отменить run {run.id} после таймаута: {cancel_error}")
            return "Извините, ответ занимает слишком много времени. Попробуйте переформулировать вопрос или повторить попытку позже."

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
            await add_message_to_history(user_id, "assistant", assistant_response) # Добавляем в локальную историю
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
            return "Ассистент обработал запрос, но не смог сформировать текстовый ответ."

    except openai.APIError as e:
         logger.error(f"OpenAI API ошибка для user_id={user_id}: {e}", exc_info=True)
         return f"Произошла ошибка при обращении к OpenAI: {e}. Попробуйте позже."
    except Exception as e:
        logger.error(f"Непредвиденная ошибка в chat_with_assistant для user_id={user_id}: {e}", exc_info=True)
        return f"Произошла внутренняя ошибка при обработке вашего запроса."


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
            return "ОШИБКА: Не удалось создать вектор для вашего запроса." # Или вернуть ""

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
            return "ОШИБКА: Не удалось выполнить поиск в базе знаний." # Или вернуть ""

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
        return "ОШИБКА: Внутренняя ошибка при поиске информации в базе знаний." # Или вернуть ""

async def update_vector_store():
    """Обновляет векторное хранилище ChromaDB документами из Google Drive."""
    logger.info("--- Запуск обновления базы знаний ---")
    if not drive_service:
        logger.error("Обновление базы знаний невозможно: сервис Google Drive не инициализирован.")
        return False
    if not vector_collection:
         logger.error("Обновление базы знаний невозможно: коллекция ChromaDB не инициализирована.")
         return False

    try:
        # 1. Читаем данные из Google Drive
        logger.info("Получение данных из Google Drive...")
        # Запускаем синхронную функцию в отдельном потоке
        documents_data = await asyncio.to_thread(read_data_from_drive)
        if not documents_data:
            logger.warning("Не найдено документов в Google Drive или произошла ошибка чтения. Обновление прервано.")
            return False
        logger.info(f"Получено {len(documents_data)} документов из Google Drive.")

        # 2. Подготовка текстов и метаданных
        all_texts = []
        all_metadatas = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Размер чанка (можно настроить)
            chunk_overlap=200, # Перекрытие чанков (можно настроить)
            length_function=len,
        )

        logger.info("Разбиение документов на чанки...")
        for doc_info in documents_data:
            doc_name = doc_info['name']
            doc_content = doc_info['content']
            if not doc_content or not doc_content.strip():
                logger.warning(f"Документ '{doc_name}' пуст или содержит только пробелы. Пропускаем.")
                continue

            try:
                chunks = text_splitter.split_text(doc_content)
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_metadatas.append({"source": doc_name, "chunk": i})
                logger.info(f"Документ '{doc_name}' разбит на {len(chunks)} чанков.")
            except Exception as e:
                logger.error(f"Ошибка при разбиении документа '{doc_name}': {e}", exc_info=True)
                continue # Пропускаем битый документ

        if not all_texts:
            logger.warning("После обработки не осталось текстовых данных для добавления в базу. Обновление прервано.")
            return False

        logger.info(f"Всего подготовлено {len(all_texts)} чанков для добавления/обновления.")

        # 3. Добавление/Обновление данных в ChromaDB
        # Рекомендуется очищать коллекцию перед полным обновлением,
        # если не требуется сохранять старые данные или если ID не стабильны.
        # Или использовать collection.upsert, если есть стабильные ID для чанков.

        logger.info(f"Очистка старых данных в коллекции '{VECTOR_DB_COLLECTION}'...")
        try:
            # Получаем все текущие ID и удаляем их
            # Это может быть неэффективно для очень больших баз
            existing_ids = await asyncio.to_thread(vector_collection.get, include=[]) # Только ID
            if existing_ids and existing_ids.get('ids'):
                 await asyncio.to_thread(vector_collection.delete, ids=existing_ids['ids'])
                 logger.info(f"Удалено {len(existing_ids['ids'])} старых записей.")
            else:
                 logger.info("Коллекция была пуста, удалять нечего.")
        except Exception as e:
             logger.error(f"Ошибка при очистке коллекции '{VECTOR_DB_COLLECTION}': {e}. Обновление может быть неполным.", exc_info=True)
             # Продолжаем попытку добавить новые данные

        logger.info(f"Добавление {len(all_texts)} новых чанков в коллекцию...")
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
               vector_collection.add,
               ids=all_ids,
               embeddings=all_embeddings,
               metadatas=all_metadatas,
               documents=all_texts # Тексты тоже сохраняем в базе
            )

            logger.info(f"Успешно добавлено {len(all_ids)} чанков в коллекцию '{VECTOR_DB_COLLECTION}'.")
            logger.info(f"Текущее количество документов в коллекции: {vector_collection.count()}")

        except openai.APIError as e:
             logger.error(f"OpenAI API ошибка при создании эмбеддингов: {e}", exc_info=True)
             # Возможно, стоит откатить изменения или пометить обновление как неуспешное
             return False
        except Exception as e:
            logger.error(f"Ошибка при добавлении данных в ChromaDB: {e}", exc_info=True)
            return False

        logger.info("--- Обновление базы знаний успешно завершено ---")
        return True

    except Exception as e:
        logger.error(f"Критическая ошибка во время обновления базы знаний: {e}", exc_info=True)
        return False


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

async def add_message_to_history(user_id: int, role: str, content: str):
    """Добавляет сообщение в локальную историю пользователя (в памяти)."""
    user_key = get_user_key(user_id)
    if user_key not in user_messages:
        user_messages[user_key] = []

    # Ограничиваем длину для экономии памяти
    max_len = 1500
    truncated_content = content[:max_len] + ('...' if len(content) > max_len else '')

    user_messages[user_key].append({
        'role': role,
        'content': truncated_content,
        'timestamp': datetime.now()
    })
    # Очистка старых сообщений при добавлении (можно вынести в фоновую задачу)
    await cleanup_old_messages_for_user(user_key)


async def cleanup_old_messages_for_user(user_key: str):
     """Удаляет сообщения старше MESSAGE_LIFETIME_DAYS для конкретного пользователя."""
     if user_key in user_messages:
        now_dt = datetime.now() # Используем импортированный класс datetime
        lifetime = timedelta(days=MESSAGE_LIFETIME_DAYS)
        original_count = len(user_messages[user_key])
        user_messages[user_key] = [
            msg for msg in user_messages[user_key]
            if now_dt - msg['timestamp'] < lifetime
        ]
        removed_count = original_count - len(user_messages[user_key])
        if removed_count > 0:
            logger.info(f"Очищена история для user_key={user_key}. Удалено {removed_count} старых сообщений.")


async def cleanup_all_old_messages():
    """Очищает старые сообщения для всех пользователей."""
    logger.info("Запуск очистки старых сообщений в памяти...")
    # Итерируемся по копии ключей, так как словарь может изменяться
    for user_key in list(user_messages.keys()):
        await cleanup_old_messages_for_user(user_key)
    logger.info("Очистка старых сообщений завершена.")

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

async def background_cleanup_task():
    """Периодически запускает задачи очистки."""
    while True:
        await asyncio.sleep(3600) # Запускать каждый час
        logger.info("Запуск периодической фоновой очистки...")
        await cleanup_all_old_messages()
        await cleanup_old_context_logs()
        # Можно добавить очистку других ресурсов, если нужно
        logger.info("Фоновая очистка завершена.")


# --- Main Event Handler ---

async def handle_new_message(event: VkBotEvent):
    """Обрабатывает новое входящее сообщение."""
    try:
        if event.from_user: # Обрабатываем только личные сообщения боту
            user_id = event.obj.message['from_id']
            peer_id = event.obj.message['peer_id'] # ИЗМЕНЕНО: Получаем peer_id из объекта сообщения
            message_text = event.obj.message['text'].strip()
            # message_id = event.obj.message['id'] # ID сообщения VK (может пригодиться)

            if not message_text: # Игнорируем пустые сообщения
                 logger.info(f"Получено пустое сообщение от user_id={user_id}. Игнорируем.")
                 return

            is_manager = user_id in MANAGER_USER_IDS or user_id == ADMIN_USER_ID

            # 1. Проверка на команды от менеджера
            if is_manager:
                command = message_text.lower()
                if command == CMD_SILENCE.lower():
                    await silence_user(peer_id)
                    await send_vk_message(peer_id, f"🤖 Режим молчания активирован для этого диалога на {MANAGER_ACTIVE_TIMEOUT_SECONDS // 3600} ч.")
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
            #     now_local_str = datetime.datetime.now(CHELYABINSK_TZ).strftime('%H:%M:%S %Z')
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

            # Блокировка для предотвращения параллельной обработки от одного юзера
            async with user_processing_locks[peer_id]:
                try:
                    # Устанавливаем статус "печатает"
                    await set_typing_activity(peer_id)

                    # Получаем ответ от ассистента
                    response_text = await chat_with_assistant(user_id, message_text)

                    # Отправляем ответ пользователю
                    await send_vk_message(peer_id, response_text)

                except Exception as e:
                    logger.error(f"Ошибка при обработке сообщения или отправке ответа для user_id={user_id}: {e}", exc_info=True)
                    await send_vk_message(peer_id, "Произошла внутренняя ошибка при обработке вашего запроса. Попробуйте позже.")
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

async def main():
    """Главная асинхронная функция запуска бота."""
    logger.info("--- Запуск VK бота ---")

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
    try:
        for event in longpoll.listen():
             if event.type == VkBotEventType.MESSAGE_NEW:
                 # Безопасно передаем задачу в основной асинхронный цикл
                 asyncio.run_coroutine_threadsafe(handle_new_message(event), async_loop)
             else:
                  logger.debug(f"[Thread] Пропускаем событие типа {event.type}")
    except Exception as e:
         logger.error(f"Ошибка в потоке Long Poll: {e}", exc_info=True)
    finally:
         logger.info("Поток Long Poll завершен.")
# ---------------------------------------------------------------------


if __name__ == "__main__":
    try:
        # Python 3.7+
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получен сигнал KeyboardInterrupt. Завершение работы...")
    except Exception as e:
         # Ловим ошибки, которые могли возникнуть до запуска asyncio.run()
         logger.critical(f"Критическая ошибка при запуске: {e}", exc_info=True)