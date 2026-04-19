"""
ingestion.py — Document parsing, chunking, metadata tagging, and vector store upsert.

Pipeline:
  1. Load all .txt / .md files from the docs/ folder
  2. Markdown-aware semantic chunking (header splitting + table preservation + overlap)
  3. Metadata extraction via regex (document_id, policy_category, effective_date, etc.)
  4. Embed with OpenAI text-embedding-3-large
  5. Upsert into a local ChromaDB collection
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "enterprise_policies"

CHUNK_OVERLAP_RATIO = 0.12          # 12% overlap between sequential chunks
MAX_CHUNK_CHARS = 1500              # approx 375 tokens at 4 chars/token

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Travel": ["travel", "trip", "hotel", "flight", "per diem", "expense", "visa", "passport"],
    "Security": ["security", "password", "mfa", "vpn", "byod", "phishing", "data classification", "breach", "incident"],
    "HR": ["hr", "leave", "absence", "conduct", "harassment", "onboarding", "offboarding", "termination"],
    "Learning": ["learning", "training", "tuition", "certification", "conference", "stipend", "lms", "learnhub"],
    "Finance": ["finance", "reimbursement", "expense", "budget", "invoice", "paycheck", "clawback"],
    "Legal": ["gdpr", "ccpa", "pci", "compliance", "legal", "regulatory", "audit"],
    "IT": ["it", "software", "shadow it", "procurement", "saas", "api", "database", "cloud"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_doc_metadata(raw_text: str, filename: str) -> Dict[str, str]:
    """Extract structured metadata from a policy document's header block."""
    meta: Dict[str, str] = {}

    # Document ID  e.g. SEC-POL-8005-V7
    doc_id_match = re.search(r"\*\*Document ID:\*\*\s*([A-Z0-9\-]+)", raw_text)
    meta["document_id"] = doc_id_match.group(1) if doc_id_match else filename

    # Effective date
    eff_date_match = re.search(r"\*\*Effective Date:\*\*\s*([\w ,]+\d{4})", raw_text)
    meta["effective_date"] = eff_date_match.group(1).strip() if eff_date_match else ""

    # Last revised
    revised_match = re.search(r"\*\*Last Revised:\*\*\s*([\w ,]+\d{4})", raw_text)
    meta["last_revised"] = revised_match.group(1).strip() if revised_match else ""

    # Policy owner
    owner_match = re.search(r"\*\*Policy Owner:\*\*\s*(.+)", raw_text)
    meta["policy_owner"] = owner_match.group(1).strip() if owner_match else ""

    # Infer policy category from filename and content
    combined = (filename + " " + raw_text[:500]).lower()
    meta["policy_category"] = _infer_category(combined)

    # Source filename
    meta["source_file"] = filename

    return meta


def _infer_category(text: str) -> str:
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(text_lower.count(kw) for kw in keywords)
    return max(scores, key=scores.get, default="General")


# ─────────────────────────────────────────────────────────────────────────────
# Table-preserving chunker
# ─────────────────────────────────────────────────────────────────────────────

def _split_preserving_tables(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    """
    Split text into chunks while keeping Markdown tables intact.
    Tables are identified by lines starting with '|'. They are always kept
    as a single chunk (or row-chunked with header repetition if very large).
    """
    overlap_chars = int(chunk_size * overlap_ratio)
    chunks: List[str] = []
    lines = text.splitlines(keepends=True)

    current_chunk: List[str] = []
    current_len = 0
    in_table = False
    table_header: Optional[str] = None
    table_lines: List[str] = []

    def flush_chunk():
        nonlocal current_chunk, current_len
        content = "".join(current_chunk).strip()
        if content:
            chunks.append(content)
        # Carry overlap into next chunk
        if current_chunk:
            overlap_text = "".join(current_chunk)[-overlap_chars:]
            current_chunk = [overlap_text]
            current_len = len(overlap_text)
        else:
            current_chunk = []
            current_len = 0

    def flush_table():
        nonlocal table_lines, table_header
        if not table_lines:
            return
        table_text = "".join(table_lines).strip()
        # If the table fits, append to current chunk or emit standalone
        if current_len + len(table_text) <= chunk_size * 2:  # tables get 2x room
            current_chunk.append("\n" + table_text + "\n")
            # Don't flush automatically; let normal flow handle it
        else:
            # Row-chunk large tables: header + each data row
            rows = [l for l in table_lines if l.strip().startswith("|")]
            if len(rows) < 2:
                chunks.append(table_text)
            else:
                header_row = rows[0]
                separator_row = rows[1] if len(rows) > 1 else ""
                for data_row in rows[2:]:
                    row_chunk = f"{header_row}{separator_row}{data_row}".strip()
                    chunks.append(row_chunk)
        table_lines = []
        table_header = None

    i = 0
    while i < len(lines):
        line = lines[i]
        is_table_line = line.strip().startswith("|")

        if is_table_line and not in_table:
            # Entering a table — flush current prose chunk first
            flush_chunk()
            in_table = True
            table_lines = [line]
            table_header = line

        elif is_table_line and in_table:
            table_lines.append(line)

        elif not is_table_line and in_table:
            # Exiting a table
            flush_table()
            in_table = False
            current_chunk.append(line)
            current_len += len(line)
            if current_len >= chunk_size:
                flush_chunk()

        else:
            # Normal prose
            current_chunk.append(line)
            current_len += len(line)
            if current_len >= chunk_size:
                flush_chunk()

        i += 1

    if in_table:
        flush_table()
    if current_chunk:
        content = "".join(current_chunk).strip()
        if content:
            chunks.append(content)

    return [c for c in chunks if len(c.strip()) > 50]


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion functions
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk_documents(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """
    Load all policy .txt/.md files, chunk them semantically, and return
    LangChain Document objects with rich metadata.
    """
    # Header-based splitter — splits on # / ## / ###
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1_header"),
            ("##", "h2_header"),
            ("###", "h3_header"),
        ],
        strip_headers=False,
    )

    all_documents: List[Document] = []
    txt_files = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt or .md files found in {docs_dir}")

    for file_path in txt_files:
        raw_text = file_path.read_text(encoding="utf-8")
        filename = file_path.name

        # Step 1: Doc-level metadata
        doc_meta = _extract_doc_metadata(raw_text, filename)

        # Step 2: Split by Markdown headers
        header_splits = header_splitter.split_text(raw_text)

        for split in header_splits:
            section_text = split.page_content
            section_meta = {**doc_meta, **split.metadata}

            # Step 3: Sub-chunk within each header section (preserving tables)
            sub_chunks = _split_preserving_tables(
                section_text,
                chunk_size=MAX_CHUNK_CHARS,
                overlap_ratio=CHUNK_OVERLAP_RATIO,
            )

            for idx, chunk_text in enumerate(sub_chunks):
                chunk_id = hashlib.md5(
                    f"{filename}:{section_meta.get('h2_header','')}:{idx}:{chunk_text[:40]}".encode()
                ).hexdigest()[:12]

                all_documents.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **section_meta,
                        "chunk_index": idx,
                        "chunk_id": chunk_id,
                    },
                ))

    return all_documents


def build_vector_store(
    documents: List[Document],
    persist_dir: Path = CHROMA_DIR,
    api_key: Optional[str] = None,
) -> Chroma:
    """Embed documents and upsert into ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )

    persist_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )
    return vector_store


def load_vector_store(
    persist_dir: Path = CHROMA_DIR,
    api_key: Optional[str] = None,
) -> Chroma:
    """Load an already-persisted ChromaDB collection."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def ingest_pipeline(docs_dir: Path = DOCS_DIR, api_key: Optional[str] = None) -> Chroma:
    """End-to-end ingestion: load → chunk → embed → store."""
    documents = load_and_chunk_documents(docs_dir)
    vector_store = build_vector_store(documents, api_key=api_key)
    return vector_store, len(documents)
