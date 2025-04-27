from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from pathlib import Path

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
                print(f"⚠️ Skipping unsupported file: {file_path.name}")
                continue

            documents = loader.load()
            all_docs.extend(documents)

        except Exception as e:
            print(f"⚠️ Error loading {file_path.name}: {e}")

    return all_docs
