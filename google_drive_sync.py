import os
import sys
import io
import pickle
import logging
import json
import asyncio
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Отладочный вывод
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Environment variables:", os.environ)

# Настройка логирования в файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/TestBot/logs/google_drive_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Если изменить эти области, удалите файл token.pickle
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
DB_DIRECTORY = "vector_store"  # Директория для хранения векторной БД

def get_google_drive_service():
    """Получает сервис Google Drive API."""
    creds = None
    
    # Файл token.pickle хранит токены доступа и обновления пользователя
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    # Если нет действительных учетных данных, позволяем пользователю войти
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Сохраняем учетные данные для следующего запуска
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def download_and_process_file(service, file_id):
    """Скачивает и обрабатывает файл из Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        
        return file_io.getvalue().decode('utf-8')
    except Exception as e:
        logger.error(f'Error downloading file: {str(e)}')
        return None

def create_vector_store(texts):
    """Создает векторное хранилище из текстов."""
    try:
        # Разбиваем тексты на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text('\n'.join(texts))

        # Создаем векторное хранилище
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=DB_DIRECTORY
        )
        vectorstore.persist()
        logger.info('Vector store created successfully')
        return vectorstore
    except Exception as e:
        logger.error(f'Error creating vector store: {str(e)}')
        return None

def sync_folder(folder_id, local_path):
    """Синхронизирует папку Google Drive с локальной папкой и создает векторное хранилище."""
    try:
        service = get_google_drive_service()
        
        # Получаем список файлов
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType)").execute()
        files = results.get('files', [])

        if not files:
            logger.info('No files found in Google Drive folder.')
            return

        # Создаем локальную папку
        os.makedirs(local_path, exist_ok=True)
        
        # Собираем все тексты
        all_texts = []
        for file in files:
            if file['mimeType'] == 'application/vnd.google-apps.document':
                logger.info(f'Processing file: {file["name"]}')
                text = download_and_process_file(service, file['id'])
                if text:
                    all_texts.append(text)

        # Создаем векторное хранилище
        if all_texts:
            vectorstore = create_vector_store(all_texts)
            if vectorstore:
                logger.info('Vector store updated successfully')
            
        logger.info('Folder sync completed successfully')
        
    except Exception as e:
        logger.error(f'Error syncing folder: {str(e)}')

def update_data():
    """Обновляет данные из Google Drive."""
    FOLDER_ID = '1RxvI5SoJzSTH-5tJgSTgaKGglcw8oVC-'
    LOCAL_PATH = 'data'
    
    sync_folder(FOLDER_ID, LOCAL_PATH)

async def update_data_periodically(interval_hours=1):
    """Периодически обновляет данные из Google Drive."""
    logger.info("Starting periodic update service")
    try:
        # Проверяем наличие необходимых файлов
        logger.info(f"Checking credentials file: {os.path.exists('credentials.json')}")
        logger.info(f"Checking token file: {os.path.exists('token.pickle')}")
        
        # Проверяем переменные окружения
        logger.info("Checking environment variables...")
        logger.info(f"OPENAI_API_KEY exists: {'OPENAI_API_KEY' in os.environ}")
        
        while True:
            try:
                logger.info(f'Starting data update at {datetime.now()}')
                update_data()
                logger.info(f'Next update scheduled in {interval_hours} hours')
                await asyncio.sleep(interval_hours * 3600)
            except Exception as e:
                logger.error(f'Error in periodic update: {str(e)}')
                await asyncio.sleep(300)
    except Exception as e:
        logger.error(f'Critical error in service: {str(e)}')
        raise

if __name__ == "__main__":
    try:
        # Создаем и запускаем цикл событий
        loop = asyncio.get_event_loop()
        loop.run_until_complete(update_data_periodically(interval_hours=1))
        loop.run_forever()
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        raise
