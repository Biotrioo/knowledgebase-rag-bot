@echo off
echo 🔥 Activating virtual environment...
cd /d C:\AI\KnowledgebaseRAGBot\knowledgebase-rag-bot
call .\venv\Scripts\activate.bat

echo 🔥 Starting document indexing...
python vectorstore_local.py

echo 🎯 Indexing completed!
pause
