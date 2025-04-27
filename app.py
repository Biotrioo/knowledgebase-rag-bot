import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import requests
import os
import subprocess

# Local Embeddings Loader
class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Initialize Vectorstore
def load_vectorstore():
    embedding = LocalEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding,
    )
    return vectorstore

# Fetch active model from LM Studio
def get_active_model():
    try:
        response = requests.get("http://localhost:1234/v1/models")
        data = response.json()
        model_id = data["data"][0]["id"]
        return model_id
    except Exception:
        return "Unknown Model (Error Connecting)"

# Connect to LM Studio (Local Server) with Ultra-Refined Prompt
def ask_local_llm(context, question, research_mode=False):
    if research_mode:
        prompt = f"""You are a highly knowledgeable Research Assistant.

Your task:
- Perform a detailed and deep research analysis based ONLY on the provided CONTEXT.
- Summarize and synthesize information across multiple documents if needed.
- If the CONTEXT does NOT contain enough information, reply exactly: "Not enough information."
- NEVER invent, assume, or hallucinate facts.
- ALWAYS maintain a clear, structured, professional tone.

Formatting:
- Start with: **Final Deep Research Answer:**
- Use Markdown bullets or numbered lists if helpful.
- Conclude with: **End of Answer.**

CONTEXT:
{context}

QUESTION:
{question}

Final Deep Research Answer:"""
    else:
        prompt = f"""You are a highly knowledgeable, neutral, and professional Research Assistant.

Your task:
- Answer the QUESTION strictly based on the provided CONTEXT.
- If the CONTEXT does NOT contain enough information, reply exactly: "Not enough information."
- NEVER invent, assume, or hallucinate facts.
- ALWAYS maintain a formal, clear, and concise academic style.

Formatting:
- Start with: **Final Answer:**
- Use Markdown bullets or numbered lists if helpful.
- Conclude with: **End of Answer.**

CONTEXT:
{context}

QUESTION:
{question}

Final Answer:"""

    response = requests.post(
        "http://localhost:1234/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "prompt": prompt,
            "max_tokens": 4096,
            "temperature": 0.2,
            "stop": ["End of Answer.", "</think>", "<|endoftext|>"],
        }
    )
    result = response.json()
    return result["choices"][0]["text"]

# Streamlit UI
def main():
    st.set_page_config(page_title="Knowledgebase RAG Bot", page_icon=":books:")
    st.title("Private Knowledgebase Assistant :books:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    active_model = get_active_model()
    st.info(f"💬 Active Model Detected: **{active_model}**")

    # Upload PDFs
    st.subheader("📤 Upload New PDF Files")
    uploaded_files = st.file_uploader("Upload your PDFs here", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("✅ Uploaded successfully! Please click 'Reindex Knowledgebase' next.")

    # Reindex Button
    st.subheader("🔄 Reindex Knowledgebase")
    if st.button("Reindex Knowledgebase"):
        with st.spinner("Rebuilding the vectorstore..."):
            result = subprocess.run(["python", "vectorstore_local.py"], capture_output=True, text=True)
            st.success("✅ Reindexing completed!")
            st.text_area("Reindexing Log:", result.stdout)

    st.divider()

    # Load Vectorstore
    vectorstore = load_vectorstore()

    # Chat Interface
    st.subheader("🔎 Ask a Question to Your Knowledgebase")

    # Display Chat History
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

    query = st.text_input("Enter your question:")
    research_mode = st.checkbox("🧠 Deep Research Mode (slower but deeper analysis)")

    if st.button("Get Answer"):
        with st.spinner("Searching your knowledgebase..."):

            retrieval_k = 5
            if research_mode:
                retrieval_k = 50  # Pull more documents for deep analysis

            docs = vectorstore.similarity_search(query, k=retrieval_k)

            context_chunks = []
            sources = set()
            total_tokens_estimate = 0
            max_tokens_limit = 8192

            for doc in docs:
                text = doc.page_content
                est_tokens = len(text.split()) * 1.3  # rough 1 word = ~1.3 tokens
                if total_tokens_estimate + est_tokens > max_tokens_limit:
                    break
                context_chunks.append(text)
                total_tokens_estimate += est_tokens
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])

            context = "\n\n".join(context_chunks)
            answer = ask_local_llm(context, query, research_mode=research_mode)

            # Save conversation
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            st.success("Answer:")
            st.markdown(answer)

            with st.expander("📄 Retrieved Context Chunks"):
                st.write(context)

            if sources:
                st.subheader("📚 Sources Used:")
                for src in sources:
                    st.write(f"- {src}")
            else:
                st.warning("No source information available for retrieved chunks.")

if __name__ == "__main__":
    main()
