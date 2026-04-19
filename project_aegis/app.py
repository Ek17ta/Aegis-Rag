"""
app.py — Streamlit frontend for Project Aegis Enterprise RAG System.

Run with:  streamlit run app.py
"""

import os
import shutil
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Aegis — Enterprise Policy RAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
DOCS_DIR.mkdir(exist_ok=True)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; letter-spacing: 2px; }
    .main-header p { margin: 0.5rem 0 0; opacity: 0.8; font-size: 0.95rem; }

    /* Answer box */
    .answer-box {
        background: #f8f9ff;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Source card */
    .source-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .source-card .doc-id { font-weight: bold; color: #0f3460; }
    .source-card .category-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 4px;
        padding: 1px 8px;
        font-size: 0.75rem;
        margin-left: 8px;
    }

    /* Pipeline steps */
    .pipeline-step {
        background: #f0f4ff;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        border-left: 3px solid #3f51b5;
    }

    /* Status badges */
    .badge-green { color: #2e7d32; background: #e8f5e9; padding: 2px 8px; border-radius: 4px; }
    .badge-blue  { color: #1565c0; background: #e3f2fd; padding: 2px 8px; border-radius: 4px; }
    .badge-orange{ color: #e65100; background: #fff3e0; padding: 2px 8px; border-radius: 4px; }

    /* Chat history */
    .user-bubble {
        background: #e3f2fd;
        border-radius: 12px 12px 2px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .assistant-bubble {
        background: #f5f5f5;
        border-radius: 12px 12px 12px 2px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
        float: left;
        clear: both;
    }
    .clearfix { clear: both; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏛️ PROJECT AEGIS</h1>
    <p>Advanced Enterprise Policy Intelligence System · Powered by RAG + GPT-4o</p>
</div>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API Key input
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Required. Get yours at platform.openai.com",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()

    # RAG settings
    st.markdown("### 🔧 RAG Pipeline Settings")
    use_hyde = st.toggle("HyDE (Hypothetical Doc Embeddings)", value=True,
                         help="Generate a fake answer to improve retrieval precision")
    use_reranker = st.toggle("Cross-Encoder Reranker", value=True,
                              help="Use ms-marco-MiniLM to rerank retrieved chunks")

    st.divider()

    # Document upload
    st.markdown("### 📂 Policy Documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .md policy files",
        type=["txt", "md"],
        accept_multiple_files=True,
        help="Upload your corporate policy documents here",
    )

    if uploaded_files:
        saved = 0
        for uf in uploaded_files:
            dest = DOCS_DIR / uf.name
            dest.write_bytes(uf.read())
            saved += 1
        st.success(f"✅ {saved} file(s) saved to docs/")

    # Show existing docs
    existing_docs = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.md"))
    if existing_docs:
        st.markdown(f"**Loaded docs ({len(existing_docs)}):**")
        for d in existing_docs:
            st.markdown(f"- `{d.name}`")

    st.divider()

    # Ingestion button
    col_ingest, col_reset = st.columns(2)
    with col_ingest:
        ingest_btn = st.button("⚡ Build Index", type="primary", use_container_width=True)
    with col_reset:
        if st.button("🗑️ Reset", use_container_width=True):
            if CHROMA_DIR.exists():
                shutil.rmtree(CHROMA_DIR)
            st.session_state.vector_store = None
            st.session_state.ingestion_done = False
            st.session_state.doc_count = 0
            st.session_state.chat_history = []
            st.rerun()

    # Ingestion status
    if st.session_state.ingestion_done:
        st.success(f"✅ Index ready · {st.session_state.doc_count} chunks")
    else:
        st.info("Upload docs and click Build Index")


# ─── Ingestion Logic ──────────────────────────────────────────────────────────
if ingest_btn:
    if not api_key_input:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()

    existing_docs = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.md"))
    if not existing_docs:
        st.error("⚠️ No policy documents found in docs/. Please upload files first.")
        st.stop()

    with st.spinner("🔄 Building vector index… this may take 30-60 seconds."):
        try:
            from ingestion import ingest_pipeline
            # Clear old index
            if CHROMA_DIR.exists():
                shutil.rmtree(CHROMA_DIR)

            vs, doc_count = ingest_pipeline(docs_dir=DOCS_DIR, api_key=api_key_input)
            st.session_state.vector_store = vs
            st.session_state.ingestion_done = True
            st.session_state.doc_count = doc_count
            st.success(f"✅ Indexed **{doc_count}** chunks from {len(existing_docs)} document(s)!")
            st.rerun()
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            st.stop()

# Load existing index on startup
if not st.session_state.ingestion_done and CHROMA_DIR.exists() and api_key_input:
    try:
        from ingestion import load_vector_store
        vs = load_vector_store(api_key=api_key_input)
        st.session_state.vector_store = vs
        st.session_state.ingestion_done = True
    except Exception:
        pass


# ─── Main Chat Interface ───────────────────────────────────────────────────────
tab_chat, tab_pipeline, tab_docs = st.tabs(["💬 Chat", "🔬 Pipeline Inspector", "📚 Documents"])

with tab_chat:
    # Example questions
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        example_cols = st.columns(3)
        examples = [
            "What is the annual L&D stipend for Individual Contributors?",
            "What must I do if I accidentally click a phishing link?",
            "Can I use personal AI tools like ChatGPT for work tasks?",
            "How does the tuition clawback clause work if I resign?",
            "What data classifications exist and how is Tier 4 data handled?",
            "What are the MFA requirements for corporate systems?",
        ]
        for i, ex in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state._pending_query = ex

    # Render chat history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("sources"):
                with st.expander(f"📄 {len(turn['sources'])} Source(s) Retrieved"):
                    for src in turn["sources"]:
                        st.markdown(f"""
<div class="source-card">
  <span class="doc-id">{src['document_id']}</span>
  <span class="category-badge">{src['policy_category']}</span>
  <br><small>{src.get('h1_header','')} {'>' if src.get('h2_header') else ''} {src.get('h2_header','')}</small>
  <br><small style="color:#666;font-style:italic">{src['snippet']}</small>
</div>
""", unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input(
        "Ask a question about company policy…",
        disabled=not st.session_state.ingestion_done,
    )

    # Handle example button clicks
    if hasattr(st.session_state, "_pending_query"):
        user_query = st.session_state._pending_query
        del st.session_state._pending_query

    if user_query:
        if not st.session_state.ingestion_done or not st.session_state.vector_store:
            st.warning("⚠️ Please build the index first (sidebar → Build Index).")
        elif not api_key_input:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        else:
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching policies and generating answer…"):
                    try:
                        from retrieval import run_rag_pipeline
                        t0 = time.time()
                        result = run_rag_pipeline(
                            query=user_query,
                            vector_store=st.session_state.vector_store,
                            api_key=api_key_input,
                            use_hyde=use_hyde,
                            use_reranker=use_reranker,
                        )
                        elapsed = time.time() - t0

                        st.markdown(result["answer"])

                        # Source expander
                        if result["sources"]:
                            with st.expander(f"📄 {len(result['sources'])} Source(s) · {elapsed:.1f}s"):
                                for src in result["sources"]:
                                    st.markdown(f"""
<div class="source-card">
  <span class="doc-id">{src['document_id']}</span>
  <span class="category-badge">{src['policy_category']}</span>
  <br><small>{src.get('h1_header','')} {'>' if src.get('h2_header') else ''} {src.get('h2_header','')}</small>
  <br><small style="color:#666;font-style:italic">{src['snippet']}</small>
</div>
""", unsafe_allow_html=True)

                        # Store in history with pipeline metadata
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "pipeline_meta": {
                                "expanded_queries": result["expanded_queries"],
                                "hyde_doc": result["hyde_doc"],
                                "category_filter": result["category_filter"],
                                "num_candidates": result["num_candidates"],
                                "elapsed": elapsed,
                            },
                        })

                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        import traceback
                        st.code(traceback.format_exc())


with tab_pipeline:
    st.markdown("### 🔬 Pipeline Transparency Inspector")
    st.markdown("This tab shows the internal steps the RAG system took for the last query.")

    if st.session_state.chat_history:
        last_turn = st.session_state.chat_history[-1]
        pm = last_turn.get("pipeline_meta", {})

        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates Retrieved", pm.get("num_candidates", "—"))
        col2.metric("Category Filter", pm.get("category_filter", "—"))
        col3.metric("Response Time", f"{pm.get('elapsed', 0):.1f}s")

        st.markdown("#### Step 1 · Query Expansions")
        for i, q in enumerate(pm.get("expanded_queries", []), 1):
            label = "📝 Original" if i == 1 else (f"🔁 Expansion {i-1}" if "?" in q else "🧪 HyDE Document")
            st.markdown(f"""<div class="pipeline-step"><b>{label}</b><br>{q}</div>""",
                        unsafe_allow_html=True)

        if pm.get("hyde_doc"):
            st.markdown("#### Step 2 · HyDE Generated Excerpt")
            st.info(pm["hyde_doc"])

        st.markdown("#### Step 3 · Retrieved Sources (after reranking)")
        for i, src in enumerate(last_turn.get("sources", []), 1):
            with st.expander(f"Rank {i} — {src['document_id']} · {src.get('h2_header', '')}"):
                st.markdown(f"**Category:** {src['policy_category']}  |  **Date:** {src.get('effective_date','')}")
                st.text(src["snippet"])
    else:
        st.info("Ask a question in the Chat tab to see pipeline internals here.")


with tab_docs:
    st.markdown("### 📚 Indexed Document Inventory")
    existing_docs = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.md"))
    if existing_docs:
        for doc_path in existing_docs:
            with st.expander(f"📄 {doc_path.name}"):
                content = doc_path.read_text(encoding="utf-8")
                # Show first 50 lines
                lines = content.splitlines()
                st.code("\n".join(lines[:60]), language="markdown")
                if len(lines) > 60:
                    st.caption(f"… {len(lines) - 60} more lines")
    else:
        st.info("No documents uploaded yet. Use the sidebar to upload policy files.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;opacity:0.5;font-size:0.8rem">
Project Aegis · Enterprise RAG System · Built with LangChain + ChromaDB + OpenAI GPT-4o
</div>
""", unsafe_allow_html=True)
