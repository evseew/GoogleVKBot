import os
import openai
import logging
import asyncio
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Загружаем переменные из .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# Проверяем, загрузились ли переменные
if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY or not ASSISTANT_ID:
    raise ValueError("❌ Ошибка: Проверь .env файл, некоторые переменные не загружены!")

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Создаем объекты бота, диспетчера и роутера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
router = Router()

# Хранение `thread_id` и истории сообщений пользователей
user_threads = {}
user_messages = {}
MEMORY_LIMIT = 100  # Увеличили с 10 до 100 сообщений
MESSAGE_LIFETIME = timedelta(days=100)  # Увеличили до 100 дней

async def cleanup_old_messages():
    """Очищает старые сообщения"""
    # Функция больше не нужна, но оставляем пустой для обратной совместимости
    pass

async def get_or_create_thread(user_id):
    """Получает или создает новый thread_id для пользователя."""
    if user_id in user_threads:
        return user_threads[user_id]

    thread = openai.beta.threads.create()
    thread_id = thread.id

    user_threads[user_id] = thread_id
    user_messages[user_id] = []
    return thread_id

async def add_message_to_history(user_id, role, content):
    """Добавляет сообщение в историю"""
    if user_id not in user_messages:
        user_messages[user_id] = []
    
    user_messages[user_id].append({
        'role': role,
        'content': content,
        'timestamp': datetime.now()  # Оставляем timestamp для возможного использования в будущем
    })
    
    # Ограничиваем только по количеству сообщений
    if len(user_messages[user_id]) > MEMORY_LIMIT:
        user_messages[user_id].pop(0)

async def get_conversation_context(user_id):
    """Получает контекст разговора"""
    if user_id not in user_messages:
        return ""
    
    context = "\nПредыдущий контекст разговора:\n"
    for msg in user_messages[user_id]:
        context += f"{msg['role']}: {msg['content']}\n"
    return context

def read_data_from_drive():
    """Читает данные из синхронизированной папки."""
    data_path = 'data'  # Путь к папке с данными
    result = []
    
    try:
        for filename in os.listdir(data_path):
            with open(os.path.join(data_path, filename), 'r') as f:
                result.append(f.read())
    except Exception as e:
        logging.error(f"Error reading data: {str(e)}")
    
    return result

async def get_relevant_context(query: str, k: int = 3) -> str:
    """Получает релевантный контекст из векторного хранилища."""
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="vector_store", 
            embedding_function=embeddings
        )
        docs = vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logging.error(f"Error getting context: {str(e)}")
        return ""

async def chat_with_assistant(user_id, user_message):
    """Отправляет сообщение ассистенту и получает ответ."""
    thread_id = await get_or_create_thread(user_id)
    
    # Получаем релевантный контекст из векторного хранилища
    relevant_context = await get_relevant_context(user_message)
    
    # Формируем контекст с релевантными данными
    context = f"Relevant context:\n{relevant_context}\n\nUser message: {user_message}"
    
    # Отправляем сообщение с контекстом
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=context
    )

    # Запускаем ассистента
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )

    # Ждем завершения обработки
    while True:
        run_status = openai.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if run_status.status in ["completed", "failed"]:
            break
        await asyncio.sleep(1)

    # Получаем ответ от ассистента
    messages = openai.beta.threads.messages.list(thread_id=thread_id)

    if messages and len(messages.data) > 0:
        response = messages.data[0].content[0].text.value
        return response

    return "Ошибка: не удалось получить ответ от ассистента."

@router.message(Command("start"))
async def start_command(message: types.Message):
    """Приветственное сообщение при старте."""
    await message.answer("👋 Здравствуйте. Как вам помочь?")

@router.message(Command("clear"))
async def clear_history(message: types.Message):
    """Очищает историю сообщений пользователя"""
    user_id = message.from_user.id
    if user_id in user_messages:
        user_messages[user_id] = []
    await message.answer("🧹 История разговора очищена!")

@router.business_message()
async def handle_message(message: types.Message):
    """Обрабатывает входящее сообщение пользователя в бизнес-чате."""
    user_id = message.from_user.id
    user_input = message.text

    logging.info(f"Получено сообщение от пользователя {user_id}: {user_input}")
    
    # Очищаем старые сообщения перед обработкой нового
    await cleanup_old_messages()
    
    response = await chat_with_assistant(user_id, user_input)
    
    logging.info(f"Отправляем ответ пользователю {user_id}: {response}")
    
    # Отправляем сообщение с business_connection_id
    await bot.send_message(
        chat_id=message.chat.id,
        text=response,
        business_connection_id=message.business_connection_id
    )

async def main():
    """Запуск бота (aiogram 3.x)"""
    dp.include_router(router)  # Подключаем router
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())  # Новый формат запуска для aiogram 3.x