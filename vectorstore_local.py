from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# Load Documents
def load_documents(folder_path):
    all_docs = []
    data_folder = Path(folder_path)
    for file_path in data_folder.glob('*'):
        try:
            if file_path.suffix == '.pdf':
                loader = UnstructuredPDFLoader(str(file_path))
            elif file_path.suffix == '.docx':
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif file_path.suffix == '.html':
                loader = UnstructuredHTMLLoader(str(file_path))
            else:
                print(f"⚠️ Unsupported file type: {file_path}")
                continue

            documents = loader.load()
            all_docs.extend(documents)

        except Exception as e:
            print(f"⚠️ Error loading {file_path.name}: {e}")
    return all_docs

# Split Documents into Chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)

# Correct Local Embeddings Class
class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Main Function
def main():
    documents = load_documents('data')
    print(f"✅ Loaded {len(documents)} documents.")
    split_docs = split_documents(documents)
    print(f"✅ Split into {len(split_docs)} chunks.")

    embedder = LocalEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedder,
        persist_directory="./chroma_db"
    )

    vectorstore.persist()
    print("✅ Vectorstore created and saved locally!")

    # 🎯 Final success message
    print("\n🎯 Ingestion and Vectorstore Creation Completed Successfully!\n")

# Ensure script runs
if __name__ == "__main__":
    main()
