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
from langchain.text_splitter import MarkdownHeaderTextSplitter

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
        
        # Создаем сплиттер заголовков Markdown
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
            ]
        )
        
        # Запасной вариант для не-Markdown документов
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Новый размер чанка
            chunk_overlap=100,  # Новое значение перекрытия
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        print("Разбиваем документы на фрагменты...")
        for doc in documents_data:
            # Добавляем имя файла в начало содержимого для лучшей идентификации
            enhanced_content = f"Документ: {doc['name']}\n\n{doc['content']}"
            
            # Определяем, является ли документ Markdown по расширению или содержимому
            is_markdown = doc['name'].endswith('.md') or '##' in doc['content'] or '#' in doc['content']
            
            if is_markdown:
                # Используем специальный сплиттер для Markdown
                try:
                    md_header_splits = markdown_splitter.split_text(enhanced_content)
                    
                    # Если документ слишком длинный, дополнительно делим каждый раздел
                    if any(len(d.page_content) > 2000 for d in md_header_splits):
                        final_docs = []
                        for md_doc in md_header_splits:
                            # Сохраняем метаданные заголовков
                            headers_metadata = {k: v for k, v in md_doc.metadata.items() if k.startswith('header_')}
                            
                            # Делим большие разделы на более мелкие части
                            smaller_chunks = text_splitter.split_text(md_doc.page_content)
                            for chunk in smaller_chunks:
                                final_docs.append(
                                    Document(
                                        page_content=chunk,
                                        metadata={
                                            "source": doc['name'],
                                            "document_type": "markdown",
                                            **headers_metadata  # Сохраняем структуру заголовков
                                        }
                                    )
                                )
                        docs.extend(final_docs)
                    else:
                        # Добавляем дополнительные метаданные и собираем документы
                        for md_doc in md_header_splits:
                            md_doc.metadata["source"] = doc['name']
                            md_doc.metadata["document_type"] = "markdown"
                        docs.extend(md_header_splits)
                        
                except Exception as e:
                    print(f"Ошибка при обработке Markdown документа {doc['name']}: {str(e)}")
                    # В случае ошибки, используем обычный сплиттер как запасной вариант
                    splits = text_splitter.split_text(enhanced_content)
                    for split in splits:
                        docs.append(
                            Document(
                                page_content=split,
                                metadata={
                                    "source": doc['name'],
                                    "document_type": "text"
                                }
                            )
                        )
            else:
                # Для обычных текстовых документов
                splits = text_splitter.split_text(enhanced_content)
                for split in splits:
                    docs.append(
                        Document(
                            page_content=split,
                            metadata={
                                "source": doc['name'],
                                "document_type": "text"
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