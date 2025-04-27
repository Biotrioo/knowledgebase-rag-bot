# 📚 Knowledgebase RAG Bot

A fully private, offline AI Research Assistant built with:

- ✅ Retrieval-Augmented Generation (RAG) architecture
- ✅ Local document ingestion (PDF, DOCX, HTML)
- ✅ Local LLM inference via LM Studio (DeepSeek, Mistral, Llama3, etc.)
- ✅ Smart Deep Research Mode (dynamic context handling)
- ✅ Streamlit frontend (for easy chat interface)

> Built to empower serious researchers, traders, analysts, and thinkers with real deep reasoning capabilities — 100% offline, 100% under your control.

---

## 🚀 Features

- 📂 Upload private documents (PDFs, DOCXs, HTMLs)
- 🔎 Ask quick questions or run full **Deep Research Mode**
- 📚 Retrieve knowledge from hundreds of pages at once
- 🧠 Dynamic context limiting (~8K prompt tokens) for stability
- 💬 Streamlit Web App (Private frontend)
- 🖥️ Runs fully local using LM Studio for models
- 🔐 No OpenAI, No API keys, No external calls — full privacy

---

## 📂 Project Structure

| Folder/File | Purpose |
|:--|:--|
| `app.py` | Streamlit frontend app |
| `vectorstore_local.py` | Indexing PDFs into local ChromaDB |
| `load_documents.py` | Document loader module |
| `chroma_db/` | (Ignored) Vector database (created locally) |
| `venv/` | (Ignored) Python virtual environment |
| `data/` | (Ignored) Folder where you upload your PDFs |

✅ `.gitignore` ensures no private files are uploaded.

---

## ⚙️ Requirements

- Python 3.10+
- LM Studio running locally (`localhost:1234`)
- Installed models (DeepSeek, Mistral, Llama, Gemma, etc.)
- Streamlit
- Sentence Transformers
- LangChain
- Unstructured (for PDF/Docx parsing)

---

## 🛠️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/Biotrioo/knowledgebase-rag-bot.git
cd knowledgebase-rag-bot
