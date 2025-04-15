import sys
import os
import time
import aiohttp
import re
import glob
import asyncio
from collections import deque
import datetime
import subprocess
import signal
import shutil

# Отладочная информация
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"Virtual env: {os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')}")
print(f"Working directory: {os.getcwd()}")

# Проверка наличия библиотеки
print("Используем OpenAI Embeddings вместо sentence-transformers")

import openai
import logging
from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.filters import Command, Filter
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain_community.vectorstores import Chroma
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO
import docx
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Загружаем переменные из .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# Добавляем константу для пути к ключу сервисного аккаунта
SERVICE_ACCOUNT_FILE = 'service-account-key.json'
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")  # ID папки с документами

# Добавим константу с ID администратора бота
ADMIN_USER_ID = 164266775  # Замените на ваш ID пользователя

# Проверяем, загрузились ли переменные
if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY or not ASSISTANT_ID:
    raise ValueError("❌ Ошибка: Проверь .env файл, некоторые переменные не загружены!")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Создаем объекты бота, диспетчера и роутера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
router = Router()

# Хранение `thread_id` и истории сообщений пользователей
user_threads = {}
user_messages = {}
MESSAGE_LIFETIME = timedelta(days=100)  # Сообщения хранятся 100 дней

# Добавить кэширование ответов
response_cache = {}

# Кэширование документов из Google Drive
drive_cache = {}

# Включаем векторную базу
USE_VECTOR_STORE = True

# Создаем директорию для логов, если она не существует
LOGS_DIR = "./logs/context_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Добавляем словарь для отслеживания активных запросов и очередей сообщений
user_processing_locks = {}
user_message_queues = {}

# Добавим словарь для отслеживания последних сообщений пользователей
user_last_message_time = {}
MESSAGE_COOLDOWN = 3  # минимальный интервал между сообщениями в секундах

# Добавляем словарь для отслеживания активных менеджеров в чатах
# Ключ: business_connection_id, Значение: {timestamp: время последнего сообщения, active: True/False}
manager_presence = {}

# Время в секундах, в течение которого бот "молчит" после появления менеджера
MANAGER_ACTIVE_TIMEOUT = 1800  # 30 минут

async def get_or_create_thread(user_id):
    """Получает или создает новый thread_id для пользователя."""
    if user_id in user_threads:
        thread_id = user_threads[user_id]
        
        # Проверяем, действителен ли thread_id
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            # Пробуем получить сообщения из треда
            client.beta.threads.messages.list(thread_id=thread_id)
            return thread_id
        except Exception as e:
            logging.error(f"Ошибка доступа к треду {thread_id}: {str(e)}")
            # Если произошла ошибка, создаем новый тред
            logging.info(f"Создаем новый тред для пользователя {user_id}")
    
    # Создаем новый тред
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    thread = client.beta.threads.create()
    thread_id = thread.id

    user_threads[user_id] = thread_id
    user_messages[user_id] = []
    return thread_id

async def cleanup_old_messages():
    """Очищает старые сообщения по времени"""
    current_time = datetime.now()
    for user_id in user_messages:
        # Создаем новый список только с сообщениями, которые не старше MESSAGE_LIFETIME
        user_messages[user_id] = [
            msg for msg in user_messages[user_id] 
            if current_time - msg['timestamp'] < MESSAGE_LIFETIME
        ]

async def add_message_to_history(user_id, role, content):
    """Добавляет сообщение в историю"""
    if user_id not in user_messages:
        user_messages[user_id] = []
    
    user_messages[user_id].append({
        'role': role,
        'content': content,
        'timestamp': datetime.now()
    })
    
    # Очищаем старые сообщения при добавлении нового
    await cleanup_old_messages()

async def get_conversation_context(user_id):
    """Получает контекст разговора"""
    if user_id not in user_messages:
        return ""
    
    context = "\nПредыдущий контекст разговора:\n"
    for msg in user_messages[user_id]:
        context += f"{msg['role']}: {msg['content']}\n"
    return context

def get_drive_service():
    """Получение сервиса Google Drive через сервисный аккаунт"""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=credentials)

def read_data_from_drive():
    """Читает данные из Google Drive"""
    global drive_cache  # Добавьте эту строку
    
    # # Проверяем, есть ли кэш и не устарел ли он
    # cache_time = os.path.getmtime("vector_store") if os.path.exists("vector_store") else 0
    # current_time = time.time()
    
    # # Если кэш свежий (менее 1 часа), используем его
    # if current_time - cache_time < 3600 and drive_cache:
    #     logging.info("Используем кэшированные данные из Google Drive") # Добавим лог
    #     return drive_cache
    
    # # Иначе загружаем данные заново
    # logging.info("Кэш устарел или отсутствует. Загружаем данные из Google Drive...") # Добавим лог
    service = get_drive_service()
    result = []
    
    try:
        # Получаем список файлов из указанной папки
        files = service.files().list(
            q=f"'{FOLDER_ID}' in parents",
            fields="files(id, name, mimeType)"
        ).execute()

        for file in files.get('files', []):
            content = ""
            file_id = file['id']
            mime_type = file['mimeType']

            try:
                if mime_type == 'application/vnd.google-apps.document':
                    # Для Google Docs
                    content = download_google_doc(service, file_id)
                elif mime_type == 'application/pdf':
                    # Для PDF файлов
                    content = download_pdf(service, file_id)
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    # Для DOCX файлов
                    content = download_docx(service, file_id)
                elif mime_type == 'text/plain':
                    # Для TXT файлов
                    content = download_text(service, file_id)

                if content:
                    result.append({
                        'name': file['name'],
                        'content': content
                    })
                    logging.info(f"Successfully read file: {file['name']}")

            except Exception as e:
                logging.error(f"Error reading file {file['name']}: {str(e)}")
                continue
            
    except Exception as e:
        logging.error(f"Error reading from Google Drive: {str(e)}")
    
    # # Сохраняем в кэш
    # drive_cache = result
    return result

def download_google_doc(service, file_id):
    """Скачивает и читает содержимое Google Doc."""
    try:
        # Экспортируем в текстовый формат
        content = service.files().export(
            fileId=file_id,
            mimeType='text/plain'
        ).execute()
        return content.decode('utf-8')
    except Exception as e:
        logging.error(f"Error downloading Google Doc: {str(e)}")
        return ""

def download_pdf(service, file_id):
    """Скачивает и читает содержимое PDF файла."""
    try:
        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO(request.execute())
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error downloading PDF: {str(e)}")
        return ""

def download_docx(service, file_id):
    """Скачивает и читает содержимое DOCX файла."""
    try:
        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO(request.execute())
        doc = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error downloading DOCX: {str(e)}")
        return ""

def download_text(service, file_id):
    """Скачивает и читает содержимое текстового файла."""
    try:
        request = service.files().get_media(fileId=file_id)
        file = io.BytesIO(request.execute())
        return file.getvalue().decode('utf-8')
    except Exception as e:
        logging.error(f"Error downloading text file: {str(e)}")
        return ""

async def get_relevant_context(query: str, k: int = 3) -> str:
    """Получает релевантный контекст из векторного хранилища."""
    try:
        persist_directory = "./local_vector_db"
        collection_name = "documents"
        
        # Проверяем существование базы данных
        if not os.path.exists(persist_directory):
            logging.error(f"База данных не найдена по пути '{persist_directory}'")
            return "ВНИМАНИЕ: База данных не найдена. Пожалуйста, обновите базу знаний с помощью команды /update."
        
        try:
            import chromadb
            from openai import OpenAI

            # Подключаемся к ChromaDB используя новый API
            logging.info(f"Подключаемся к базе данных для запроса: '{query}'")
            chroma_client = chromadb.PersistentClient(
                path=persist_directory
            )
            
            # Проверяем доступность коллекций
            try:
                collections = chroma_client.list_collections()
                collection_names = [c.name for c in collections]
                logging.info(f"Доступные коллекции: {collection_names}")
                
                if collection_name not in collection_names:
                    logging.error(f"Коллекция '{collection_name}' не найдена!")
                    return "ВНИМАНИЕ: База данных не содержит нужную коллекцию. Пожалуйста, обновите базу знаний с помощью команды /update."
                
            except Exception as avail_err:
                logging.error(f"Ошибка при проверке доступных коллекций: {str(avail_err)}")
                return f"ОШИБКА: Не удалось проверить наличие коллекций: {str(avail_err)}"
            
            # Получаем коллекцию
            try:
                collection = chroma_client.get_collection(name=collection_name)
                
                # Проверяем, есть ли документы в коллекции
                count = collection.count()
                if count == 0:
                    logging.error("База векторов пуста! Вызовите /update")
                    return "ВНИМАНИЕ: База данных пуста. Пожалуйста, обновите базу знаний с помощью команды /update."
                
                logging.info(f"В базе найдено {count} записей")
                
            except Exception as coll_e:
                logging.error(f"Не удалось получить коллекцию: {str(coll_e)}")
                return f"ОШИБКА: Не удалось подключиться к базе данных: {str(coll_e)}"
            
            # Создаем embedding для запроса
            client = OpenAI()  # По умолчанию использует OPENAI_API_KEY из переменных окружения
            model_name = "text-embedding-3-large"
            embed_dim = 1536
            
            # Получаем вектор запроса
            logging.info(f"Получаем вектор для запроса: '{query}'")
            query_embedding_response = client.embeddings.create(
                input=[query],
                model=model_name,
                dimensions=embed_dim
            )
            query_embedding = query_embedding_response.data[0].embedding
            
            # Выполняем поиск в коллекции
            logging.info(f"Выполняем поиск по запросу: '{query}'")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,  # Запрашиваем больше результатов для фильтрации
                include=["documents", "metadatas", "distances"]
            )
            
            # Проверяем результаты
            if not results["ids"] or not results["ids"][0] or len(results["ids"][0]) == 0:
                logging.warning(f"Не найдено релевантных документов для запроса: '{query}'")
                return ""
            
            # Формируем результаты в читаемом формате
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Создаем список кортежей (документ, метаданные, расстояние)
            doc_tuples = list(zip(documents, metadatas, distances))
            
            # Сортируем результаты по релевантности (по возрастанию расстояния)
            sorted_docs = sorted(doc_tuples, key=lambda x: x[2])
            
            # Подробное логирование каждого найденного документа
            logging.info(f"Найдено {len(sorted_docs)} документов для запроса '{query}'")
            for i, (doc_text, metadata, distance) in enumerate(sorted_docs):
                logging.info(f"Документ #{i+1}:")
                logging.info(f"Источник: {metadata.get('source', 'неизвестно')}")
                logging.info(f"Релевантность (расстояние): {distance:.4f}")
                logging.info(f"Содержание: {doc_text[:200]}...")
            
            # Берем только k наиболее релевантных документов
            top_docs = sorted_docs[:k]
            
            # Формируем контекст с указанием источников
            context_pieces = []
            for doc_text, metadata, distance in top_docs:
                source = metadata.get('source', 'неизвестный источник')
                context_pieces.append(f"Из документа '{source}':\n{doc_text}")
                
            found_text = "\n\n".join(context_pieces)
            
            if not found_text:
                logging.warning(f"Не найдено релевантных документов для запроса: '{query}'")
            return found_text
            
        except ImportError as ie:
            logging.error(f"Не установлена необходимая библиотека: {str(ie)}")
            return f"ОШИБКА: Не установлены библиотеки для работы с базой данных: {str(ie)}"
            
    except Exception as e:
        logging.error(f"Ошибка при получении контекста: {str(e)}")
        # Возвращаем информацию об ошибке в контексте
        return f"ОШИБКА: Не удалось получить данные из базы знаний: {str(e)}"

async def cleanup_old_context_logs():
    """Удаляет логи контекста, которые старше 24 часов"""
    try:
        current_time = time.time()
        one_day_ago = current_time - 86400  # 24 часа в секундах
        
        # Ищем все файлы контекста в папке логов
        log_files = glob.glob(os.path.join(LOGS_DIR, "context_log_*_*.txt"))
        
        count = 0
        for log_file in log_files:
            file_mod_time = os.path.getmtime(log_file)
            if file_mod_time < one_day_ago:
                os.remove(log_file)
                count += 1
        
        logging.info(f"Очистка логов: удалено {count} устаревших файлов контекста")
    except Exception as e:
        logging.error(f"Ошибка при очистке логов контекста: {str(e)}")

async def chat_with_assistant(user_id, message_text):
    """Отправляет сообщение ассистенту и получает ответ"""
    try:
        # Очищаем старые логи перед созданием новых
        asyncio.create_task(cleanup_old_context_logs())
        
        # Получаем thread_id для пользователя или создаем новый
        thread_id = await get_or_create_thread(user_id)
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Добавляем отладочные сообщения
        logging.info(f"Запрос контекста для: '{message_text}'")
        
        # Получаем контекст из векторного хранилища
        if USE_VECTOR_STORE:
            context = await get_relevant_context(message_text)
            logging.info(f"Получен контекст длиной {len(context)} символов")
            if context:
                # Логируем контекст для анализа
                await log_context(user_id, message_text, context)
            else:
                logging.warning("Контекст не найден в базе")
        else:
            context = ""
        
        # Формируем полный запрос с контекстом
        full_prompt = f"Контекст: {context}\n\nВопрос: {message_text}"
        
        # Проверяем, нет ли активного запроса для этого треда
        try:
            # Получаем список запусков для этого треда
            runs = client.beta.threads.runs.list(thread_id=thread_id)
            
            # Проверяем, есть ли активные запуски
            active_runs = [run for run in runs.data if run.status in ['queued', 'in_progress', 'requires_action']]
            
            if active_runs:
                # Отменяем все активные запуски
                for run in active_runs:
                    try:
                        client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                        logging.info(f"Отменен активный запуск {run.id} для треда {thread_id}")
                    except Exception as cancel_error:
                        logging.warning(f"Не удалось отменить запуск {run.id}: {str(cancel_error)}")
        except Exception as list_runs_error:
            logging.warning(f"Ошибка при проверке активных запусков: {str(list_runs_error)}")
        
        # Добавляем сообщение пользователя в тред
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=full_prompt
        )
        
        # Запускаем запрос к ассистенту
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        # Ждем завершения запроса с таймаутом
        max_wait_time = 60  # максимальное время ожидания в секундах
        start_time = time.time()

        while True:
            # Проверяем, не превышено ли время ожидания
            if time.time() - start_time > max_wait_time:
                logging.warning(f"Превышено время ожидания ответа от ассистента для запроса: {message_text}")
                return "Извините, мне нужно немного больше времени для ответа. Можете повторить ваш вопрос?"
            
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                return f"Извините, не удалось обработать ваш запрос. Попробуйте, пожалуйста, еще раз."
            await asyncio.sleep(1)
        
        # Получаем ответ ассистента
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        
        # Берем самое новое сообщение от ассистента
        for msg in messages.data:
            if msg.role == "assistant":
                response = msg.content[0].text.value
                # Сохраняем сообщение в истории
                await add_message_to_history(user_id, "user", message_text)
                await add_message_to_history(user_id, "assistant", response)
                return response
                
        return "Ассистент не смог сформировать ответ."
        
    except Exception as e:
        logging.error(f"Error in chat_with_assistant: {str(e)}")
        return f"Произошла ошибка: {str(e)}"

async def update_vector_store():
    """Обновляет векторное хранилище документами из Google Drive"""
    try:
        # Получаем данные из Drive
        logging.info("Начинаем обновление: получаем данные из Google Drive...")
        documents_data = read_data_from_drive()
        if not documents_data:
            logging.warning("Не получено данных из Google Drive. Обновление прервано.")
            return True
        logging.info(f"Получено {len(documents_data)} документов из Google Drive.")

        # Подготавливаем документы для индексации
        docs = []
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "header_1"),("##", "header_2"),("###", "header_3"),("####", "header_4"),]
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, # Используем увеличенные значения
            chunk_overlap=200, # Используем увеличенные значения
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        for doc_data in documents_data:
            content_str = doc_data.get('content', '')
            if not isinstance(content_str, str):
                logging.warning(f"Содержимое документа {doc_data.get('name', 'N/A')} не является строкой, пропускаем.")
                continue
            enhanced_content = f"Документ: {doc_data.get('name', 'N/A')}\n\n{content_str}"
            doc_name = doc_data.get('name', 'unknown')
            is_markdown = doc_name.endswith('.md') or '##' in content_str or '#' in content_str
            try:
                if is_markdown:
                    try:
                        md_header_splits = markdown_splitter.split_text(enhanced_content)
                        if any(len(d.page_content) > 2000 for d in md_header_splits):
                            final_docs_part = []
                            for md_doc in md_header_splits:
                                headers_metadata = {k: v for k, v in md_doc.metadata.items() if k.startswith('header_')}
                                smaller_chunks = text_splitter.split_text(md_doc.page_content)
                                for chunk in smaller_chunks:
                                    final_docs_part.append(Document(page_content=chunk, metadata={"source": doc_name, "document_type": "markdown", **headers_metadata}))
                            docs.extend(final_docs_part)
                        else:
                            for md_doc in md_header_splits:
                                md_doc.metadata["source"] = doc_name
                                md_doc.metadata["document_type"] = "markdown"
                            docs.extend(md_header_splits)
                    except Exception as e_md:
                        logging.error(f"Ошибка при обработке Markdown документа {doc_name}: {str(e_md)}. Пробуем как обычный текст.")
                        splits = text_splitter.split_text(enhanced_content)
                        for split in splits:
                             docs.append(Document(page_content=split, metadata={"source": doc_name, "document_type": "text_fallback"}))
                else:
                    splits = text_splitter.split_text(enhanced_content)
                    for split in splits:
                        docs.append(Document(page_content=split, metadata={"source": doc_name, "document_type": "text"}))
            except Exception as e_doc:
                 logging.error(f"Не удалось обработать документ {doc_name}: {str(e_doc)}")
                 continue
        # --- КОНЕЦ ОБРАБОТКИ ДОКУМЕНТОВ ---

        if not docs:
            logging.warning("После обработки документов не осталось чанков для добавления в базу.")
            return True

        logging.info(f"Подготовлено {len(docs)} чанков для добавления в базу.")

        # --- НАЧАЛО РАБОТЫ С CHROMADB ---
        try:
            import chromadb
            from openai import OpenAI

            client = OpenAI() 
            model_name = "text-embedding-3-large"
            embed_dim = 1536
            
            # --- ОБНОВЛЕНИЕ КОНФИГУРАЦИИ ДЛЯ ПОДДЕРЖКИ НОВОГО API CHROMADB ---
            persist_directory = "./local_vector_db"
            logging.info(f"Создаем директорию для базы данных: {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)
            
            # Создаем клиент ChromaDB по новому API
            logging.info("Создаем клиент ChromaDB...")
            chroma_client = chromadb.PersistentClient(
                path=persist_directory
            )
            
            # Очищаем существующую коллекцию, если она есть
            collection_name = "documents"
            try:
                logging.info(f"Проверяем наличие и удаляем существующую коллекцию '{collection_name}'...")
                collections = chroma_client.list_collections()
                if any(c.name == collection_name for c in collections):
                    chroma_client.delete_collection(name=collection_name)
                    logging.info(f"Удалена существующая коллекция '{collection_name}'")
            except Exception as e_coll:
                logging.warning(f"Ошибка при проверке/удалении коллекции: {str(e_coll)}")
            
            # Создаем новую коллекцию
            logging.info(f"Создаем новую коллекцию '{collection_name}'...")
            collection = chroma_client.create_collection(name=collection_name)
            
            # Подготавливаем данные для добавления
            batch_size = 40  # Меньший размер партии для лучшей стабильности
            total_added = 0
            
            # Обрабатываем документы партиями
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                chunk_ids = []
                chunk_texts = []
                chunk_metadatas = []
                
                # Подготовка данных для текущей партии
                for j, doc in enumerate(batch):
                    doc_id = f"doc_{i+j}"
                    chunk_ids.append(doc_id)
                    chunk_texts.append(doc.page_content)
                    chunk_metadatas.append(doc.metadata)
                
                # Получаем встраивания от OpenAI для текущей партии
                logging.info(f"Получаем встраивания для партии {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} ({len(chunk_texts)} документов)...")
                try:
                    embeddings_response = client.embeddings.create(
                        input=chunk_texts,
                        model=model_name,
                        dimensions=embed_dim
                    )
                    chunk_embeddings = [e.embedding for e in embeddings_response.data]
                    
                    # Добавляем текущую партию
                    collection.add(
                        ids=chunk_ids,
                        documents=chunk_texts,
                        metadatas=chunk_metadatas,
                        embeddings=chunk_embeddings
                    )
                    
                    total_added += len(chunk_texts)
                    logging.info(f"Добавлено {total_added}/{len(docs)} документов.")
                    
                except Exception as e:
                    logging.error(f"Критическая ошибка при обработке партии документов {i//batch_size + 1}: {str(e)}", exc_info=True)
                    # Прерываем весь процесс при любой ошибке
                    return False
            
            # Проверяем, что мы добавили все документы успешно
            if total_added == len(docs):
                logging.info(f"Обновление векторного хранилища успешно завершено. Добавлено {total_added} документов.")
                # Записываем время обновления в файл 
                save_vector_db_creation_time()
                return True
            else:
                logging.warning(f"Обновление завершено, но добавлено только {total_added}/{len(docs)} документов!")
                return False
            
        except ImportError as ie:
            logging.error(f"Не установлена необходимая библиотека: {str(ie)}")
            return False
        except Exception as e:
            logging.error(f"Критическая ошибка при создании векторного хранилища: {str(e)}", exc_info=True)
            return False

    except Exception as e:
        logging.error(f"Критическая ошибка при обновлении векторного хранилища: {str(e)}", exc_info=True)
        return False

@router.message(Command("start"))
async def start_command(message: types.Message):
    """Приветственное сообщение при старте."""
    await message.answer("👋 Здравствуйте! Подождите, я обновляю базу знаний...")
    
    # Обновляем векторное хранилище
    success = await update_vector_store()
    
    if success:
        await message.answer("✅ База знаний обновлена! Как я могу вам помочь?")
    else:
        await message.answer("❌ Произошла ошибка при обновлении базы знаний. Но вы можете задавать вопросы!")

@router.message(Command("clear"))
async def clear_history(message: types.Message):
    """Очищает историю сообщений пользователя"""
    user_id = message.from_user.id
    if user_id in user_messages:
        user_messages[user_id] = []
    await message.answer("🧹 История разговора очищена!")

@router.message(Command("reset"))
async def reset_conversation(message: types.Message):
    """Полностью сбрасывает разговор, включая удаление треда и создание нового"""
    user_id = message.from_user.id
    
    # Очищаем историю сообщений
    if user_id in user_messages:
        user_messages[user_id] = []
    
    # Удаляем тред, чтобы создать новый при следующем сообщении
    if user_id in user_threads:
        del user_threads[user_id]
    
    await message.answer("🔄 Разговор полностью сброшен! Ваш следующий вопрос начнет новый диалог.")

@router.message(Command("reset_all"))
async def reset_all_conversations(message: types.Message):
    """Полностью сбрасывает все разговоры всех пользователей (только для администратора)"""
    user_id = message.from_user.id
    
    # Проверяем, имеет ли пользователь права администратора
    if user_id != ADMIN_USER_ID:
        await message.answer("❌ У вас нет прав для выполнения этой команды!")
        return
    
    # Очищаем историю сообщений всех пользователей
    user_messages.clear()
    
    # Удаляем треды всех пользователей
    user_threads.clear()
    
    # Сообщаем об успешном сбросе
    await message.answer("🔄 Все разговоры всех пользователей полностью сброшены!")
    logging.info(f"Администратор {user_id} выполнил полный сброс всех диалогов")

@router.message(Command("update"))
async def update_knowledge(message: types.Message):
    """Обновляет базу знаний вручную"""
    await message.answer("🔄 Обновляю базу знаний...")
    success = await update_vector_store()
    
    if success:
        await message.answer("✅ База знаний успешно обновлена!")
    else:
        await message.answer("❌ Произошла ошибка при обновлении базы знаний")

@router.message(Command("check_db"))
async def check_database(message: types.Message):
    """Проверяет наличие векторной базы знаний."""
    if os.path.exists("./local_vector_db"):
        files = os.listdir("./local_vector_db")
        await message.answer(f"✅ База знаний существует!\nФайлы: {', '.join(files)}")
    else:
        await message.answer("❌ База знаний не найдена.")

@router.message(Command("debug_db"))
async def debug_database(message: types.Message):
    """Диагностика базы данных векторов."""
    try:
        await message.answer("🔍 Проверяю базу векторов...")
        
        # Проверяем наличие директории
        if not os.path.exists("./local_vector_db"):
            await message.answer("❌ Директория базы не существует!")
            return
        
        # Получаем время создания базы
        db_time = get_vector_db_creation_time()
        if db_time:
            time_str = db_time.strftime("%d.%m.%Y %H:%M:%S")
            await message.answer(f"📅 Время последнего обновления: {time_str}")
        
        # Проверяем файлы в директории
        files = os.listdir("./local_vector_db")
        await message.answer(f"📂 Файлы в директории базы: {', '.join(files)}")
        
        # Пробуем создать и проверить работу базы
        try:
            # Используем единую размерность 1536
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                dimensions=1536  # Единая размерность для всех операций
            )
            
            vector_store = Chroma(
                collection_name="documents",
                embedding_function=embeddings,
                persist_directory="./local_vector_db"
            )
            
            # Получаем количество записей
            collection = vector_store.get()
            count = len(collection['ids'])
            await message.answer(f"📊 Количество записей в базе: {count}")
            
            # Тестовый запрос к базе
            docs = vector_store.similarity_search("тестовый запрос")
            
            # Успешно!
            await message.answer("✅ База векторов работает корректно")
            
        except Exception as e:
            await message.answer(f"❌ Ошибка диагностики: {str(e)}")
            
    except Exception as e:
        await message.answer(f"❌ Ошибка диагностики: {str(e)}")

@router.message(Command("db_time"))
async def check_db_time(message: types.Message):
    """Показывает время последнего обновления базы данных"""
    db_time = get_vector_db_creation_time()
    
    if db_time:
        time_str = db_time.strftime("%d.%m.%Y %H:%M:%S")
        await message.answer(f"📅 База данных последний раз обновлялась: {time_str}")
    else:
        await message.answer("❌ Не удалось определить время обновления базы данных или база не существует")

@router.message(Command("full_debug"))
async def full_debug(message: types.Message):
    """Полная диагностика системы и базы данных"""
    try:
        await message.answer("🔎 Запускаю полную диагностику системы...")
        
        # Текущая директория
        current_dir = os.getcwd()
        await message.answer(f"📂 Текущая рабочая директория: {current_dir}")
        
        # Проверяем все возможные пути базы
        db_paths = [
            "./local_vector_db",
            "/Users/test/Documents/GoogleBusinessBot/local_vector_db",
            f"{current_dir}/local_vector_db"
        ]
        
        for path in db_paths:
            if os.path.exists(path):
                await message.answer(f"✅ Путь существует: {path}")
                # Проверка содержимого
                files = os.listdir(path)
                if files:
                    await message.answer(f"📄 Содержимое: {', '.join(files[:10])}...")
                    # Проверка размера директории
                    total_size = sum(os.path.getsize(os.path.join(path, f)) for f in files if os.path.isfile(os.path.join(path, f)))
                    await message.answer(f"📊 Общий размер: {total_size/1024/1024:.2f} МБ")
                    
                    # Проверка времени модификации
                    try:
                        latest_mod = max(os.path.getmtime(os.path.join(path, f)) for f in files if os.path.isfile(os.path.join(path, f)))
                        mod_time = datetime.fromtimestamp(latest_mod)
                        await message.answer(f"🕒 Последнее изменение: {mod_time.strftime('%d.%m.%Y %H:%M:%S')}")
                    except Exception as e:
                        await message.answer(f"❌ Ошибка при получении времени: {str(e)}")
                else:
                    await message.answer(f"⚠️ Путь {path} пуст!")
            else:
                await message.answer(f"❌ Путь не существует: {path}")
        
        # Проверка, где файлы создаются при обновлении
        await message.answer("🔄 Проверяю, где создаются файлы при обновлении...")
        # Смотрим path в методе update_vector_store
        from inspect import getsource
        
        # Попытка найти базу данных везде
        await message.answer("🔍 Поиск базы данных на всём диске...")
        import subprocess
        try:
            result = subprocess.run(["find", "/", "-name", "chroma.sqlite3", "-type", "f"], 
                                  capture_output=True, text=True, timeout=10)
            if result.stdout:
                paths = result.stdout.strip().split("\n")
                await message.answer(f"🔎 Найдены SQLite файлы Chroma: {', '.join(paths)}")
            else:
                await message.answer("❌ Файлы базы данных не найдены")
        except Exception as e:
            await message.answer(f"❌ Ошибка при поиске: {str(e)}")
            
    except Exception as e:
        await message.answer(f"❌ Ошибка диагностики: {str(e)}")

@router.message(Command("debug_context"))
async def debug_context(message: types.Message):
    """Показывает контекст, который модель получает для последнего запроса"""
    user_id = message.from_user.id
    query = message.text.replace("/debug_context", "").strip()
    
    if not query:
        # Если запрос не указан в команде, просим его ввести
        await message.answer("Введите запрос после команды, например: `/debug_context как обучаться?`")
        return
    
    await message.answer(f"🔍 Получаю контекст для запроса: '{query}'...")
    
    # Получаем контекст из векторного хранилища
    context = await get_relevant_context(query)
    
    # Если контекст слишком длинный, разбиваем его на части
    max_length = 4000  # Максимальная длина сообщения в Telegram
    
    if not context:
        await message.answer("❌ Контекст не найден для данного запроса.")
        return
        
    if len(context) <= max_length:
        await message.answer(f"📚 Контекст для запроса '{query}':\n\n{context}")
    else:
        parts = [context[i:i+max_length] for i in range(0, len(context), max_length)]
        await message.answer(f"📚 Контекст для запроса '{query}' (разбит на {len(parts)} частей):")
        
        for i, part in enumerate(parts):
            await message.answer(f"Часть {i+1}/{len(parts)}:\n\n{part}")
            # Добавляем небольшую задержку между сообщениями
            await asyncio.sleep(1)

@dp.business_message()
async def handle_business_message(message: types.Message):
    """Обрабатывает входящее сообщение пользователя в бизнес-чате."""
    user_id = message.from_user.id
    user_input = message.text
    business_connection_id = message.business_connection_id

    logging.info(f"Получено бизнес-сообщение от пользователя {user_id}: {user_input}")
    logging.info(f"Business connection ID: {business_connection_id}")
    
    # Расширенная проверка сообщения от менеджера
    is_from_business = False
    
    # Метод 1: Стандартный атрибут
    if hasattr(message, 'is_from_business'):
        is_from_business = message.is_from_business
        logging.info(f"Атрибут is_from_business: {is_from_business}")
    
    # Метод 2: Проверка наличия атрибута from_business_id
    if hasattr(message, 'from_business_id') and message.from_business_id:
        is_from_business = True
        logging.info(f"Найден from_business_id: {message.from_business_id}")
    
    # Метод 3: Проверка метаданных сообщения
    if hasattr(message, 'via_bot') and message.via_bot:
        is_from_business = True
        logging.info(f"Сообщение отправлено через бизнес-бота: {message.via_bot}")
    
    # Метод 4: Просто в целях отладки - вывести все атрибуты сообщения
    for attr_name in dir(message):
        if not attr_name.startswith('_') and not callable(getattr(message, attr_name)):
            try:
                attr_value = getattr(message, attr_name)
                if attr_value and 'business' in str(attr_name).lower():
                    logging.info(f"Обнаружен бизнес-атрибут: {attr_name} = {attr_value}")
            except:
                pass
    
    # Временное решение: Ручной режим - определить менеджера по ключевым словам
    manager_keywords = ["стоимость", "цена", "оплата", "скидк", "акци"]
    if any(keyword in user_input.lower() for keyword in manager_keywords):
        logging.info(f"Обнаружено ключевое слово менеджера в сообщении: {user_input}")
        is_from_business = True
    
    if is_from_business:
        # Это сообщение от менеджера - активируем режим "молчания" бота
        logging.info(f"Обнаружено сообщение от менеджера бизнеса в чате {business_connection_id}")
        await set_manager_active(business_connection_id, True)
        return  # Бот не отвечает на сообщения менеджеров
    
    # Проверяем, активен ли менеджер в данном чате
    if await is_manager_active(business_connection_id):
        logging.info(f"Бот в режиме молчания в чате {business_connection_id}, т.к. менеджер активен")
        return  # Если менеджер активен, бот не отвечает на сообщения клиента
    
    # Проверяем, не слишком ли часто пользователь отправляет сообщения
    current_time = time.time()
    if user_id in user_last_message_time:
        time_since_last_message = current_time - user_last_message_time[user_id]
        if time_since_last_message < MESSAGE_COOLDOWN:
            await bot.send_message(
                chat_id=message.chat.id,
                text="⚠️ Пожалуйста, подождите немного между сообщениями. Обрабатываю ваш предыдущий запрос...",
                business_connection_id=message.business_connection_id
            )
            return
    
    # Обновляем время последнего сообщения
    user_last_message_time[user_id] = current_time
    
    # Очищаем старые сообщения перед обработкой нового
    await cleanup_old_messages()
    
    # Получаем ответ напрямую, без очереди
    response = await chat_with_assistant(user_id, user_input)
    
    logging.info(f"Отправляем ответ пользователю {user_id}: {response}")
    
    # Отправляем сообщение с business_connection_id
    await bot.send_message(
        chat_id=message.chat.id,
        text=response,
        business_connection_id=message.business_connection_id
    )

@router.message(F.business_connection_id.is_(None))
async def handle_message(message: types.Message):
    """Обрабатывает входящее сообщение пользователя."""
    # Обычные сообщения теперь обрабатываются здесь, а бизнес-сообщения - в другом обработчике
    # Убираем проверку на бизнес-сообщения, так как они обрабатываются другим хендлером
    
    user_id = message.from_user.id
    user_input = message.text

    logging.info(f"Получено обычное сообщение от пользователя {user_id}: {user_input}")
    
    # Проверяем, не слишком ли часто пользователь отправляет сообщения
    current_time = time.time()
    if user_id in user_last_message_time:
        time_since_last_message = current_time - user_last_message_time[user_id]
        if time_since_last_message < MESSAGE_COOLDOWN:
            await message.answer(f"⚠️ Пожалуйста, подождите немного между сообщениями. Обрабатываю ваш предыдущий запрос...")
            return
    
    # Обновляем время последнего сообщения
    user_last_message_time[user_id] = current_time
    
    # Очищаем старые сообщения перед обработкой нового
    await cleanup_old_messages()
    
    # Добавляем сообщение в очередь пользователя
    if user_id not in user_message_queues:
        user_message_queues[user_id] = deque()
        # Если это первое сообщение, создаем задачу для обработки очереди
        asyncio.create_task(process_user_message_queue(user_id))
    
    # Добавляем сообщение в очередь
    user_message_queues[user_id].append(message)

async def process_user_message_queue(user_id):
    """Обрабатывает очередь сообщений пользователя"""
    # Отмечаем, что началась обработка
    user_processing_locks[user_id] = True
    
    try:
        # Обрабатываем сообщения, пока очередь не пуста
        while user_id in user_message_queues and user_message_queues[user_id]:
            # Берем первое сообщение из очереди
            message = user_message_queues[user_id][0]
            user_input = message.text
            
            try:
                # Получаем ответ
                response = await chat_with_assistant(user_id, user_input)
                
                # Проверяем, является ли сообщение бизнес-сообщением
                if message.business_connection_id:
                    # Отправляем ответ с business_connection_id
                    await bot.send_message(
                        chat_id=message.chat.id,
                        text=response,
                        business_connection_id=message.business_connection_id
                    )
                else:
                    # Отправляем обычный ответ
                    await message.answer(response)
                
            except Exception as e:
                logging.error(f"Ошибка при обработке сообщения: {str(e)}")
                # Не отправляем сообщение об ошибке пользователю
            
            # Удаляем обработанное сообщение из очереди
            user_message_queues[user_id].popleft()
    
    finally:
        # Удаляем блокировку после завершения обработки всех сообщений
        if user_id in user_processing_locks:
            del user_processing_locks[user_id]
        
        # Если очередь пуста, можно очистить её
        if user_id in user_message_queues and not user_message_queues[user_id]:
            del user_message_queues[user_id]

# Обновленный класс для работы с бизнес-функциями
class BusinessFeatures:
    def __init__(self, bot):
        self.bot = bot
        self.connected_businesses = {}  # Хранит информацию о подключенных бизнес-аккаунтах
    
    async def get_business_info(self, business_id):
        """Получает информацию о подключенном бизнес-аккаунте"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getBusinessInfo"
            data = {"business_id": business_id}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    result = await response.json()
                    
                    if result.get("ok"):
                        business_info = result.get("result")
                        # Сохраняем информацию в кэше
                        self.connected_businesses[business_id] = business_info
                        return business_info
                    return None
        except Exception as e:
            logging.error(f"Ошибка при получении информации о бизнесе: {str(e)}")
            return None
    
    async def set_business_hours(self, business_id, hours):
        """Устанавливает часы работы для бизнес-аккаунта"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setBusinessHours"
            data = {
                "business_id": business_id,
                "hours": hours  # Формат: [{"day_of_week": 1, "start_time": "09:00", "end_time": "18:00"}, ...]
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return await response.json()
        except Exception as e:
            logging.error(f"Ошибка при установке часов работы: {str(e)}")
            return None
    
    async def set_greeting_message(self, business_id, message_text, language_code="ru"):
        """Устанавливает приветственное сообщение для бизнес-аккаунта"""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setBusinessGreetingMessage"
            data = {
                "business_id": business_id,
                "message": message_text,
                "language_code": language_code
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return await response.json()
        except Exception as e:
            logging.error(f"Ошибка при установке приветственного сообщения: {str(e)}")
            return None

async def periodic_cleanup():
    """Запускает периодическую очистку логов"""
    while True:
        try:
            await cleanup_old_context_logs()
            # Запускаем очистку раз в час
            await asyncio.sleep(3600)
        except Exception as e:
            logging.error(f"Ошибка в периодической очистке: {str(e)}")
            await asyncio.sleep(60)  # Подождем минуту перед следующей попыткой

def save_vector_db_creation_time():
    """Сохраняет текущее время как время создания/обновления векторной базы данных"""
    try:
        # Путь к файлу, где будем хранить время
        timestamp_file = os.path.join("./local_vector_db", "last_update.txt")
        
        # Записываем текущее время
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(timestamp_file, "w") as f:
            f.write(current_time)
        
        logging.info(f"Сохранено время обновления базы данных: {current_time}")
        return True
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении времени обновления базы: {str(e)}")
        return False

def get_vector_db_creation_time():
    """Получает время создания/обновления векторной базы данных"""
    try:
        # Сначала проверяем файл с сохраненным временем
        timestamp_file = os.path.join("./local_vector_db", "last_update.txt")
        if os.path.exists(timestamp_file):
            try:
                with open(timestamp_file, "r") as f:
                    time_str = f.read().strip()
                    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except Exception as file_err:
                logging.warning(f"Ошибка при чтении файла времени обновления: {str(file_err)}")
                # Продолжаем с методом получения времени из модификации файлов
        
        # Проверяем оба возможных пути
        possible_paths = [
            "./local_vector_db",
            "/Users/test/Documents/GoogleBusinessBot/local_vector_db",
            os.path.join(os.getcwd(), "local_vector_db")
        ]
        
        for db_path in possible_paths:
            logging.info(f"Проверяем путь к базе данных: {db_path}")
            if os.path.exists(db_path):
                logging.info(f"Найдена директория базы данных: {db_path}")
                
                # Проверяем файл базы данных SQLite для Chroma
                chroma_db_file = os.path.join(db_path, "chroma.sqlite3")
                if os.path.exists(chroma_db_file):
                    mod_time = os.path.getmtime(chroma_db_file)
                    return datetime.fromtimestamp(mod_time)
                
                # Получаем список всех файлов в директории
                files = [os.path.join(db_path, f) for f in os.listdir(db_path) 
                         if os.path.isfile(os.path.join(db_path, f))]
                
                if not files:
                    logging.warning(f"Директория {db_path} существует, но не содержит файлов")
                    continue
                    
                # Получаем время модификации каждого файла
                mod_times = [os.path.getmtime(f) for f in files]
                
                # Находим самое позднее время модификации
                latest_time = max(mod_times)
                
                # Исправляем ошибку здесь - было datetime.datetime.fromtimestamp(latest_time)
                return datetime.fromtimestamp(latest_time)
        
        logging.error("Не найдена директория базы данных ни по одному из проверяемых путей")
        return None
        
    except Exception as e:
        logging.error(f"Ошибка при получении времени создания базы: {str(e)}")
        return None

async def log_context(user_id, query, context):
    """Логирует запрос и контекст в отдельный файл"""
    try:
        timestamp = int(time.time())
        filename = f"context_log_{user_id}_{timestamp}.txt"
        filepath = os.path.join(LOGS_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Запрос: {query}\n\n")
            f.write(f"Контекст:\n{context}\n")
    except Exception as e:
        logging.error(f"Ошибка при логировании контекста: {str(e)}")

async def is_manager_active(business_connection_id):
    """Проверяет, активен ли менеджер в данном чате"""
    if business_connection_id not in manager_presence:
        return False
    
    # Возвращаем статус активности - бот молчит, пока менеджер не будет явно деактивирован
    # командой /unsilence
    return manager_presence[business_connection_id]["active"]

async def set_manager_active(business_connection_id, active=True):
    """Устанавливает статус активности менеджера в чате"""
    manager_presence[business_connection_id] = {
        "timestamp": time.time(),
        "active": active
    }
    logging.info(f"Менеджер {'активен' if active else 'неактивен'} в чате {business_connection_id}")

@router.message(Command("silence"))
async def silence_bot(message: types.Message):
    """Включает режим молчания бота для текущего чата (только для менеджеров)."""
    # Проверка, что команда вызвана в бизнес-чате
    if not message.business_connection_id:
        await message.answer("❌ Эта команда доступна только в бизнес-чатах!")
        return
    
    # Проверка, что команду вызвал менеджер (или администратор)
    is_from_business = getattr(message, 'is_from_business', False)
    is_admin = message.from_user.id == ADMIN_USER_ID
    
    if not (is_from_business or is_admin):
        logging.warning(f"Пользователь {message.from_user.id} попытался использовать команду /silence, но он не является менеджером")
        return  # Тихо игнорируем, если не менеджер
    
    # Включаем режим молчания
    await set_manager_active(message.business_connection_id, True)
    await message.answer("🔇 Бот переведен в режим молчания. Он не будет отвечать клиентам, пока вы общаетесь в чате.")

@router.message(Command("unsilence"))
async def unsilence_bot(message: types.Message):
    """Выключает режим молчания бота для текущего чата (только для менеджеров)."""
    # Проверка, что команда вызвана в бизнес-чате
    if not message.business_connection_id:
        await message.answer("❌ Эта команда доступна только в бизнес-чатах!")
        return
    
    # Проверка, что команду вызвал менеджер (или администратор)
    is_from_business = getattr(message, 'is_from_business', False)
    is_admin = message.from_user.id == ADMIN_USER_ID
    
    if not (is_from_business or is_admin):
        logging.warning(f"Пользователь {message.from_user.id} попытался использовать команду /unsilence, но он не является менеджером")
        return  # Тихо игнорируем, если не менеджер
    
    # Выключаем режим молчания
    if message.business_connection_id in manager_presence:
        manager_presence[message.business_connection_id]["active"] = False
        await message.answer("🔊 Режим молчания отключен. Бот снова будет отвечать на сообщения клиентов.")
    else:
        await message.answer("ℹ️ Бот уже находится в активном режиме для этого чата.")

async def main():
    """Основная функция запуска бота."""
    logging.info("🚀 Запуск бота...")
    
    # Создаем PID файл
    create_pid_file()
    
    # Регистрируем обработчики сигналов для корректного завершения
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Проверяем подключение к Google Drive
        logging.info("📁 Проверка подключения к Google Drive...")
        service = get_drive_service()
        logging.info("✅ Подключение к Google Drive успешно")
        
        # Регистрируем роутер
        dp.include_router(router)
        
        # Запускаем задачу периодической очистки логов
        asyncio.create_task(periodic_cleanup())
        
        logging.info("🤖 Бот готов к работе")
        logging.info("💼 Базовая поддержка бизнес-режима")
        
        # Запускаем бота
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"❌ Ошибка при запуске бота: {str(e)}")
        raise

def create_pid_file():
    """Создает PID файл для текущего процесса."""
    pid = os.getpid()
    # Проверяем существующие PID файлы
    if os.path.exists('bot.pid'):
        # Если основной PID файл существует, создаем пронумерованный
        i = 2
        while os.path.exists(f'bot {i}.pid'):
            i += 1
        pid_file = f'bot {i}.pid'
    else:
        pid_file = 'bot.pid'
    
    # Записываем PID в файл
    with open(pid_file, 'w') as f:
        f.write(str(pid))
    logging.info(f"Создан PID файл: {pid_file}")

def signal_handler(sig, frame):
    """Обработчик сигналов для корректного завершения работы."""
    logging.info(f"Получен сигнал завершения работы: {sig}")
    
    # Удаляем PID файлы
    if os.path.exists('bot.pid'):
        try:
            os.remove('bot.pid')
            logging.info("Удален файл bot.pid")
        except Exception as e:
            logging.error(f"Ошибка при удалении bot.pid: {str(e)}")
    
    # Удаляем другие PID файлы
    for pid_file in glob.glob('bot *.pid'):
        try:
            os.remove(pid_file)
            logging.info(f"Удален файл {pid_file}")
        except Exception as e:
            logging.error(f"Ошибка при удалении {pid_file}: {str(e)}")
    
    logging.info("Бот корректно завершил работу")
    sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Бот остановлен")
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")
        # В случае неожиданной ошибки, удаляем PID файлы
        for pid_file in glob.glob('bot*.pid'):
            try:
                os.remove(pid_file)
            except:
                pass