import os
import shutil
import asyncio
import logging
from dotenv import load_dotenv

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

# Импортируем необходимые компоненты из bot.py
from bot import read_data_from_drive
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

async def rebuild_db_fixed():
    """Пересоздает векторную базу с исправленной функцией"""
    print("Запускаем обновление базы с исправленной функцией...")
    
    try:
        # Получаем данные из Drive
        print("Загружаем документы из Google Drive...")
        documents_data = read_data_from_drive()
        print(f"Загружено {len(documents_data)} документов")
        
        # Подготавливаем документы для индексации
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # ~300 токенов - меньший размер для более точных ответов
            chunk_overlap=400,  # ~100 токенов - 33% перекрытие для поддержания контекста
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],  # Разделение по структуре маркдаун
            length_function=len
        )
        
        print("Разбиваем документы на фрагменты...")
        for doc in documents_data:
            # Добавляем имя файла в начало содержимого для лучшей идентификации
            enhanced_content = f"Документ: {doc['name']}\n\n{doc['content']}"
            splits = text_splitter.split_text(enhanced_content)
            for split in splits:
                docs.append(
                    Document(
                        page_content=split,
                        metadata={
                            "source": doc['name'],
                            "document_type": "markdown" if doc['name'].endswith('.md') else "text"
                        }
                    )
                )
        print(f"Создано {len(docs)} фрагментов текста")
        
        # Создаем векторное хранилище
        print("Удаляем старую базу данных...")
        if os.path.exists("./local_vector_db"):
            shutil.rmtree("./local_vector_db")
        
        print("Создаем новую векторную базу...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536  # Увеличиваем размерность для лучшего качества
        )
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./local_vector_db",
            collection_name="documents"
        )
        
        # Проверяем, что база данных создалась корректно
        collection = vectorstore.get()
        if len(collection['ids']) == 0:
            print("❌ Ошибка: база данных пуста!")
            return False
        print(f"✓ База данных создана и содержит {len(collection['ids'])} записей")
        
        print("✅ База данных успешно обновлена!")
        print(f"Добавлено {len(docs)} фрагментов из {len(documents_data)} документов")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обновлении базы: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(rebuild_db_fixed()) 