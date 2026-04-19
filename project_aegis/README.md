# 🏛️ Project Aegis — Enterprise Policy RAG System

A production-ready, deployable RAG chatbot for corporate policy documents. Built with LangChain, ChromaDB, and GPT-4o.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
# Option A: .env file (recommended)
cp .env.example .env
# Edit .env and add your key: OPENAI_API_KEY=sk-...

# Option B: Environment variable
export OPENAI_API_KEY=sk-...
```

### 3. Add your policy documents
Place `.txt` or `.md` policy files into the `docs/` folder.
```bash
cp your_policy.txt docs/
```

The sample policies included are:
- `it_security_and_data_privacy.txt`
- `learning_and_tuition.txt`

### 4. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│          QUERY TRANSFORMATION           │
│  • LLM Intent Router → Category Filter  │
│  • Multi-Query Expansion (3 variants)   │
│  • HyDE (Hypothetical Doc Embedding)    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           VECTOR RETRIEVAL              │
│  • ChromaDB (text-embedding-3-large)   │
│  • Metadata Pre-Filter by Category     │
│  • Top-K = 20 candidates pooled        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           POST-PROCESSING               │
│  • Date-Based Deduplication            │
│  • Cross-Encoder Reranking             │
│    (ms-marco-MiniLM-L-6-v2)           │
│  • Top-5 final chunks selected         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         ANSWER GENERATION               │
│  • GPT-4o with cited policy excerpts   │
│  • Structured source attribution       │
└─────────────────────────────────────────┘
```

---

## ☁️ Deployment Options

### Option A: Streamlit Community Cloud (Free)
1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the entry point
4. Add `OPENAI_API_KEY` in **Secrets** (Settings → Secrets)

### Option B: Railway / Render (Free tier)
1. Push to GitHub
2. Create a new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add `OPENAI_API_KEY` as an environment variable

### Option C: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t project-aegis .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... project-aegis
```

---

## 📁 Project Structure

```
project_aegis/
├── app.py              # Streamlit UI (main entry point)
├── ingestion.py        # Document parsing, chunking, embedding, ChromaDB upsert
├── retrieval.py        # Full RAG pipeline (expand → retrieve → rerank → generate)
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── docs/               # Drop your .txt/.md policy files here
│   ├── it_security_and_data_privacy.txt
│   └── learning_and_tuition.txt
└── chroma_db/          # Auto-created on first index build (gitignore this)
```

---

## 🔑 Key Features

| Feature | Implementation |
|---|---|
| Semantic chunking | MarkdownHeaderTextSplitter + table preservation |
| Token overlap | 12% overlap between sequential chunks |
| Metadata extraction | Regex → document_id, category, effective_date |
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector store | ChromaDB (local, persistent) |
| Query expansion | GPT-4o generates 3 alternate phrasings |
| HyDE | GPT-4o writes a fake policy excerpt for retrieval |
| Intent routing | GPT-4o classifies query → metadata pre-filter |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | GPT-4o with strict citation format |

---

## 📝 Adding More Documents

1. Place `.txt` or `.md` files in `docs/`
2. Ensure the file has a standard header:
   ```
   **Document ID:** YOUR-POL-ID-V1
   **Effective Date:** January 1, 2026
   **Policy Owner:** Your Department
   ```
3. Click **Build Index** in the sidebar to re-ingest

---

## ⚠️ Notes
- The `chroma_db/` folder is auto-generated and should be added to `.gitignore`
- Re-clicking "Build Index" clears and rebuilds the entire index
- The cross-encoder model downloads ~85MB on first run (cached after that)
- All data stays local — nothing is sent to external services except OpenAI API calls
