"""
retrieval.py — Advanced retrieval pipeline implementing:
  1. Query Transformation  — Multi-Query Expansion + HyDE
  2. Metadata Pre-Filtering — LLM-based intent routing to category filter
  3. Post-Filtering        — Keep only most-recent effective date per doc family
  4. Reranking             — Cross-encoder scoring via sentence-transformers
  5. Answer Generation     — OpenAI GPT-4o with structured citations
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_RETRIEVAL = 20       # Broad retrieval before reranking
TOP_K_RERANKED = 5         # Final chunks passed to the LLM
EXPANSION_COUNT = 3        # Number of query expansions

POLICY_CATEGORIES = ["Travel", "Security", "HR", "Learning", "Finance", "Legal", "IT", "General"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Query Transformation
# ─────────────────────────────────────────────────────────────────────────────

def expand_query(query: str, llm: ChatOpenAI) -> List[str]:
    """
    Multi-Query Expansion: generate EXPANSION_COUNT alternative phrasings.
    Returns the original query plus the expansions.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at reformulating questions about corporate policy documents. "
         "Your task is to rewrite the user's question in {n} different ways to maximize "
         "the chance of finding the relevant policy text in a vector database. "
         "Return ONLY a numbered list, one question per line, no explanations."),
        ("human", "Original question: {query}\n\nGenerate {n} alternative phrasings:"),
    ])
    chain = prompt | llm
    response = chain.invoke({"query": query, "n": EXPANSION_COUNT})
    raw = response.content.strip()
    expansions = [
        re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        for line in raw.splitlines()
        if line.strip() and re.match(r"^\d", line.strip())
    ]
    return [query] + expansions[:EXPANSION_COUNT]


def generate_hyde(query: str, llm: ChatOpenAI) -> str:
    """
    HyDE: generate a hypothetical policy document excerpt that would answer
    the query, then use its embedding for retrieval.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a corporate policy writer. Write a short, realistic policy document "
         "excerpt (2-4 sentences) that DIRECTLY answers the following question. "
         "Use formal policy language with specific numbers/rules. "
         "Do NOT say you don't know — always write a plausible excerpt."),
        ("human", "Question: {query}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return response.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Intent Routing & Metadata Filtering
# ─────────────────────────────────────────────────────────────────────────────

def detect_intent_category(query: str, llm: ChatOpenAI) -> Optional[str]:
    """
    Use LLM to classify the query intent into a policy category for pre-filtering.
    Returns None if the intent is ambiguous/cross-category.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the following employee question into the SINGLE most relevant "
         "corporate policy category, or say 'General' if it spans multiple. "
         f"Categories: {', '.join(POLICY_CATEGORIES)}. "
         "Reply with ONLY the category name, nothing else."),
        ("human", "{query}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"query": query})
    category = response.content.strip()
    return category if category in POLICY_CATEGORIES else None


def _retrieve_for_query(
    query: str,
    vector_store: Chroma,
    category_filter: Optional[str],
    k: int,
) -> List[Document]:
    """Run a single retrieval query with optional metadata pre-filter."""
    search_kwargs: Dict[str, Any] = {"k": k}
    if category_filter and category_filter != "General":
        search_kwargs["filter"] = {"policy_category": category_filter}

    try:
        results = vector_store.similarity_search(query, **search_kwargs)
    except Exception:
        # Fall back without filter if ChromaDB filter fails
        results = vector_store.similarity_search(query, k=k)

    return results


def retrieve_candidates(
    queries: List[str],
    vector_store: Chroma,
    category_filter: Optional[str],
    k: int = TOP_K_RETRIEVAL,
) -> List[Document]:
    """
    Retrieve candidates for all query expansions + HyDE, deduplicate by chunk_id.
    """
    seen_ids = set()
    candidates: List[Document] = []

    for q in queries:
        results = _retrieve_for_query(q, vector_store, category_filter, k=k // len(queries) + 2)
        for doc in results:
            cid = doc.metadata.get("chunk_id", doc.page_content[:40])
            if cid not in seen_ids:
                seen_ids.add(cid)
                candidates.append(doc)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Post-Filtering — keep most recent version per document family
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date_rough(date_str: str) -> str:
    """Return a sortable string from date fields like 'June 1, 2026'."""
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    date_lower = date_str.lower()
    year_match = re.search(r"\d{4}", date_str)
    year = year_match.group() if year_match else "0000"
    month = "00"
    for m_name, m_num in months.items():
        if m_name in date_lower:
            month = m_num
            break
    return f"{year}-{month}"


def post_filter_by_date(candidates: List[Document]) -> List[Document]:
    """
    Among chunks from the same document_id family (same prefix before version),
    keep only those from the most recent effective_date.
    """
    if not candidates:
        return candidates

    # Group by document family (strip version suffix e.g. -V3 → base id)
    family_map: Dict[str, List[Document]] = {}
    for doc in candidates:
        raw_id = doc.metadata.get("document_id", "unknown")
        family = re.sub(r"-V\d+$", "", raw_id, flags=re.IGNORECASE)
        family_map.setdefault(family, []).append(doc)

    filtered: List[Document] = []
    for family, docs in family_map.items():
        # Find the max effective date in this family
        best_date = max(
        _parse_date_rough(d.metadata.get("effective_date", "")) for d in docs
        )
        for doc in docs:
            doc_date = _parse_date_rough(doc.metadata.get("effective_date", ""))
            if doc_date == best_date or not doc.metadata.get("effective_date"):
                filtered.append(doc)

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Reranking (Cross-Encoder)
# ─────────────────────────────────────────────────────────────────────────────

_reranker_model = None

def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker_model = None
    return _reranker_model


def rerank_candidates(
    query: str,
    candidates: List[Document],
    top_k: int = TOP_K_RERANKED,
) -> List[Document]:
    """
    Score candidates with a cross-encoder and return top_k.
    Falls back to first top_k if model is unavailable.
    """
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_k]

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """You are an authoritative corporate policy assistant for enterprise employees.
Your job is to answer questions STRICTLY based on the provided policy document excerpts.

Rules:
- Answer directly and concisely. Lead with the key fact/number/rule.
- Cite the source policy for every claim using [Policy Name, Section] format.
- If the answer contains numbers (dollar amounts, days, percentages), bold them.
- If you cannot find a definitive answer in the excerpts, say so clearly — never invent policy.
- Structure your answer with short paragraphs. Use bullet points only for lists of rules.
- At the end, add a "📄 Sources" section listing the document IDs referenced.
"""

def generate_answer(
    query: str,
    context_docs: List[Document],
    llm: ChatOpenAI,
) -> Tuple[str, List[Dict]]:
    """
    Generate a cited answer from the top reranked context chunks.
    Returns (answer_text, list_of_source_metadata).
    """
    if not context_docs:
        return "I could not find relevant policy information for your query.", []

    # Build context block
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        meta = doc.metadata
        header = f"[Excerpt {i}] {meta.get('document_id', 'Unknown')} — {meta.get('h1_header', '')} > {meta.get('h2_header', '')}"
        context_parts.append(f"{header}\n{doc.page_content}")

    context_text = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human",
         "Policy Excerpts:\n\n{context}\n\n"
         "Employee Question: {question}\n\n"
         "Provide a comprehensive, cited answer:"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": query})

    sources = [
        {
            "document_id": d.metadata.get("document_id", "Unknown"),
            "source_file": d.metadata.get("source_file", ""),
            "policy_category": d.metadata.get("policy_category", ""),
            "h1_header": d.metadata.get("h1_header", ""),
            "h2_header": d.metadata.get("h2_header", ""),
            "effective_date": d.metadata.get("effective_date", ""),
            "snippet": d.page_content[:200] + "…",
        }
        for d in context_docs
    ]

    return response.content, sources


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(
    query: str,
    vector_store: Chroma,
    api_key: Optional[str] = None,
    use_hyde: bool = True,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: transform query → retrieve → post-filter → rerank → generate.

    Returns a dict with:
      - answer: str
      - sources: List[Dict]
      - expanded_queries: List[str]
      - hyde_doc: str
      - category_filter: str
      - num_candidates: int
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )

    # 1. Intent routing
    category_filter = detect_intent_category(query, llm)

    # 2. Query expansion
    expanded_queries = expand_query(query, llm)

    # 3. HyDE
    hyde_doc = ""
    if use_hyde:
        hyde_doc = generate_hyde(query, llm)
        expanded_queries.append(hyde_doc)

    # 4. Retrieval
    candidates = retrieve_candidates(expanded_queries, vector_store, category_filter)

    # 5. Post-filter by date
    candidates = post_filter_by_date(candidates)

    # 6. Reranking
    if use_reranker:
        final_docs = rerank_candidates(query, candidates, top_k=TOP_K_RERANKED)
    else:
        final_docs = candidates[:TOP_K_RERANKED]

    # 7. Answer generation
    answer, sources = generate_answer(query, final_docs, llm)

    return {
        "answer": answer,
        "sources": sources,
        "expanded_queries": expanded_queries[:EXPANSION_COUNT + 1],
        "hyde_doc": hyde_doc,
        "category_filter": category_filter or "None (cross-category)",
        "num_candidates": len(candidates),
    }
