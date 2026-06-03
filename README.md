# RAG Chatbot 📄🤖

Transform your PDFs into an intelligent knowledge base. Ask anything. Get instant answers powered by AI.

---

## Features ✨

• Upload and process PDF documents
• Ask natural language questions about document content
• Semantic search powered by FAISS vector store
• AI responses from Google Gemini
• Clean, interactive Streamlit UI

---

## Tech Stack 🛠️

• Python
• Streamlit (UI)
• LangChain (LLM orchestration)
• FAISS (Vector search)
• Google Gemini (Embeddings & generation)
• PDFPlumber (PDF text extraction)

---

## Quick Start ⚙️

1. **Clone and setup**
   ```bash
   git clone <your-repo-url>
   cd PythonProject
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   - Edit `env.py` and add your Google Gemini API key:
   ```python
   GEMINI_API_KEY = "your-api-key-here"
   ```

3. **Run the app**
   ```bash
   streamlit run ragchatbot.py
   ```

4. **Use it**
   - Upload a PDF file
   - Type your question
   - Get instant answers powered by your document

---

## How It Works 🔄

PDF → Text Extraction → Chunking → Embeddings → Vector Store → Retrieval → AI Response

The app extracts text from your PDF, breaks it into chunks, creates vector embeddings with Google Gemini, stores them in FAISS, and retrieves relevant sections to answer your questions.

---

## Requirements

- Python 3.11+
- Google Gemini API key (get it [here](https://ai.google.dev/))
