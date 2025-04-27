@echo off
echo 🔥 Activating virtual environment...
cd /d C:\AI\KnowledgebaseRAGBot\knowledgebase-rag-bot
call .\venv\Scripts\activate.bat

echo 🔥 Launching Streamlit app...
streamlit run app.py

pause
