"""
Enterprise RAG core engine using LangChain LCEL.

Loads enterprise policy PDFs with structure-aware chunking, embeds into Chroma,
and answers with a strict context-only system prompt to reduce hallucinations.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import json
from collections import Counter
from functools import lru_cache
from operator import itemgetter
from pathlib import Path

# Keep model artifacts inside the project for CLI, tests, and the web API alike.
_BASE_DIR = Path(__file__).resolve().parent
_MODEL_CACHE_DIR = _BASE_DIR / ".model_cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_MODEL_CACHE_DIR))

import pdfplumber
from dotenv import load_dotenv
from openai import APIStatusError, BadRequestError
from pydantic import BaseModel, Field
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kb_prompts import STRICT_GROUNDED_QA_SYSTEM

# Project paths (policy PDF and Chroma persist directory live next to this module)
# User-provided PDF names (compay_policy.pdf is the local filename in this project).
_POLICY_PDF_CANDIDATES = ("compay_policy.pdf", "company_policy.pdf")
_CHROMA_DIR = _BASE_DIR / "chroma_db"


def chroma_index_fingerprint() -> str:
    """
    Fingerprint of PDF + embedding settings used to build Chroma.

    When this changes, the persisted vector store must be rebuilt.
    """
    path = _policy_file()
    if not path.exists():
        return "missing"
    stat = path.stat()
    emb_backend = (os.getenv("RAG_EMBEDDING_BACKEND") or "openai").strip().lower()
    emb_model = (
        os.getenv("HF_EMBEDDING_MODEL", "")
        or os.getenv("OPENAI_EMBEDDING_MODEL", "")
        or "default"
    )
    return "|".join(
        [
            path.name,
            str(stat.st_mtime_ns),
            str(stat.st_size),
            emb_backend,
            emb_model,
            str(_chunk_size()),
            str(_chunk_overlap()),
            str(_extract_tables_enabled()),
        ]
    )


def chroma_index_is_stale() -> bool:
    """True if there is no index or the on-disk index predates the current PDF/settings."""
    if not _CHROMA_DIR.exists():
        return True
    try:
        entries = [p for p in _CHROMA_DIR.iterdir() if p.name != ".index_fingerprint"]
        if not entries:
            return True
    except OSError:
        return True
    fp_file = _CHROMA_DIR / ".index_fingerprint"
    if not fp_file.exists():
        return True
    return fp_file.read_text(encoding="utf-8").strip() != chroma_index_fingerprint()


def _write_chroma_fingerprint() -> None:
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (_CHROMA_DIR / ".index_fingerprint").write_text(
        chroma_index_fingerprint(),
        encoding="utf-8",
    )


def _policy_file() -> Path:
    """
    Resolve the enterprise policy PDF on disk.

    Override with RAG_POLICY_FILE (filename under project root or absolute path).
    Otherwise pick the first existing file among known candidates.
    """
    override = (os.getenv("RAG_POLICY_FILE") or "").strip()
    if override:
        path = Path(override)
        return path if path.is_absolute() else _BASE_DIR / path

    for name in _POLICY_PDF_CANDIDATES:
        path = _BASE_DIR / name
        if path.exists():
            return path

    # Default expected local name (user's Apple Business Conduct PDF).
    return _BASE_DIR / _POLICY_PDF_CANDIDATES[0]


def _policy_source_name() -> str:
    return _policy_file().name

# Semantic chunking defaults (override via RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP).
_CHUNK_SIZE = 900
_CHUNK_OVERLAP = 80
# Split priority: paragraph → line → CJK/EN sentence end → word → char (no mid-sentence cuts).
_SEMANTIC_SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]

# Lines matching these patterns are treated as page numbers / running headers / footers.
_PAGE_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^\s*第\s*\d{1,4}\s*页(?:\s*/\s*\d{1,4})?\s*$"),
    re.compile(r"^\s*Page\s+\d{1,4}(?:\s+of\s+\d{1,4})?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d{1,4}\s*-\s*$"),
    re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
)

# Smaller default reranker loads in seconds; set RAG_RERANKER_MODEL=BAAI/bge-reranker-base for max quality.
_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_AUDIT_REPORT_SCHEMA_JSON = (
    '{"records":[{"policy_topic":"string","rule_summary":"string",'
    '"amount_limit":"string","exceptions":"string","source_page":"string"}]}'
)


class AuditRecord(BaseModel):
    """Single compliance rule extracted for a given audit topic."""

    policy_topic: str = Field(
        ...,
        description='政策主题，如 "礼品收受" 或 "打车报销"。',
    )
    rule_summary: str = Field(
        ...,
        description="规则核心总结，描述具体合规红线。",
    )
    amount_limit: str = Field(
        ...,
        description='金额限制。若文档未明确限制，填写 "N/A"。',
    )
    exceptions: str = Field(
        ...,
        description="豁免或例外情况。若无明确例外，填写 N/A。",
    )
    source_page: str = Field(
        ...,
        description="规则依据所在页码，可使用单页或页码范围表示。",
    )


class AuditReport(BaseModel):
    """Structured audit checklist generated from retrieved policy chunks."""

    records: list[AuditRecord] = Field(
        default_factory=list,
        description="围绕审计主题抽取的合规规则清单。",
    )


def _load_env() -> None:
    """Load environment variables from a local .env file if present."""
    load_dotenv(_BASE_DIR / ".env")


def _optional_str(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _make_embeddings() -> Embeddings:
    """
    Build the embedding model.

    OpenAIEmbeddings is the default. Many OpenAI-compatible chat hosts (for
    example DeepSeek) do not expose a working /v1/embeddings route for the
    same credentials; in that case set RAG_EMBEDDING_BACKEND=hf to use a small
    local HuggingFace model (downloads weights on first run).

    Optional overrides for OpenAI-style embeddings only:
    - OPENAI_EMBEDDING_BASE_URL: embeddings base URL (falls back to OPENAI_BASE_URL)
    - OPENAI_EMBEDDING_API_KEY: embeddings API key (falls back to OPENAI_API_KEY)
    """
    backend = (os.getenv("RAG_EMBEDDING_BACKEND") or "openai").strip().lower()
    if backend in ("hf", "huggingface"):
        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HF_EMBEDDING_MODEL",
                # Multilingual default: English PDF + Chinese user questions.
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            encode_kwargs={"normalize_embeddings": True},
        )

    emb_base = _optional_str("OPENAI_EMBEDDING_BASE_URL") or _optional_str(
        "OPENAI_BASE_URL"
    )
    emb_key = _optional_str("OPENAI_EMBEDDING_API_KEY") or _optional_str(
        "OPENAI_API_KEY"
    )
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=emb_key,
        base_url=emb_base,
    )


def get_embeddings() -> Embeddings:
    """Public accessor for the embedding model used with Chroma."""
    return _make_embeddings()


def _chunk_size() -> int:
    raw = os.getenv("RAG_CHUNK_SIZE", str(_CHUNK_SIZE))
    try:
        return max(200, int(raw))
    except ValueError:
        return _CHUNK_SIZE


def _chunk_overlap() -> int:
    raw = os.getenv("RAG_CHUNK_OVERLAP", str(_CHUNK_OVERLAP))
    try:
        return max(0, int(raw))
    except ValueError:
        return _CHUNK_OVERLAP


def _extract_tables_enabled() -> bool:
    """Whether to run pdfplumber's comparatively expensive table detector."""
    return (os.getenv("RAG_EXTRACT_TABLES") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_page_artifact_line(line: str) -> bool:
    """Return True if a line is likely a page number or running header/footer."""
    stripped = line.strip()
    if not stripped:
        return True
    if len(stripped) <= 2 and stripped.isdigit():
        return True
    return any(pattern.match(stripped) for pattern in _PAGE_ARTIFACT_PATTERNS)


def _detect_repeated_margin_lines(pages_lines: list[list[str]]) -> set[str]:
    """
    Find short lines repeated across many pages (typical headers/footers).

    Only inspects the first/last two lines of each page to avoid stripping
    legitimate repeated phrases inside body text.
    """
    counts: Counter[str] = Counter()
    page_count = len(pages_lines)
    if page_count == 0:
        return set()

    for lines in pages_lines:
        if not lines:
            continue
        margin_candidates = lines[:2] + lines[-2:]
        seen_on_page: set[str] = set()
        for line in margin_candidates:
            normalized = line.strip()
            if not normalized or len(normalized) > 120:
                continue
            if normalized in seen_on_page:
                continue
            seen_on_page.add(normalized)
            counts[normalized] += 1

    threshold = max(2, int(page_count * 0.4))
    return {line for line, freq in counts.items() if freq >= threshold}


def _clean_page_text(
    raw_text: str,
    *,
    repeated_margin_lines: set[str],
) -> str:
    """Remove page artifacts, repeated headers/footers, and excessive whitespace."""
    lines = raw_text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if _is_page_artifact_line(stripped):
            continue
        if stripped in repeated_margin_lines:
            continue
        cleaned.append(stripped)

    # Collapse blank lines; normalize internal spacing on non-empty lines.
    normalized: list[str] = []
    prior_blank = False
    for line in cleaned:
        if not line:
            if not prior_blank:
                normalized.append("")
            prior_blank = True
            continue
        prior_blank = False
        normalized.append(re.sub(r"[ \t]{2,}", " ", line))

    return "\n".join(normalized).strip()


def _table_rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Render a pdfplumber table as a GitHub-flavored Markdown table."""
    if not rows:
        return ""

    cleaned_rows: list[list[str]] = []
    for row in rows:
        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
        if any(cells):
            cleaned_rows.append(cells)
    if not cleaned_rows:
        return ""

    width = max(len(row) for row in cleaned_rows)
    padded = [row + [""] * (width - len(row)) for row in cleaned_rows]
    header = padded[0]
    body = padded[1:] if len(padded) > 1 else []

    def _row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [_row(header), "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend(_row(row) for row in body)
    return "\n".join(lines)


def _looks_like_markdown_table(block: str) -> bool:
    """Heuristic: blocks produced by _table_rows_to_markdown or pipe-heavy layouts."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    pipe_rows = sum(1 for ln in lines if ln.count("|") >= 2)
    return pipe_rows >= 2 and any("---" in ln for ln in lines)


def _load_pdf_pages() -> list[Document]:
    """
    Parse the policy PDF page-by-page with pdfplumber.

    Text and tables are extracted separately so tables can be kept intact during
    chunking. Each page becomes a Document with ``source`` and ``page`` metadata.
    """
    policy_path = _policy_file()
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Policy PDF not found: {policy_path}. "
            f"Place one of {_POLICY_PDF_CANDIDATES} in the project root, "
            "or set RAG_POLICY_FILE in .env."
        )

    page_texts: list[str] = []
    page_line_lists: list[list[str]] = []
    page_tables: list[list[list[list[str | None]]]] = []

    extract_tables = _extract_tables_enabled()
    with pdfplumber.open(policy_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            page_texts.append(raw)
            page_line_lists.append(raw.splitlines())
            page_tables.append(page.extract_tables() or [] if extract_tables else [])

    repeated_margin_lines = _detect_repeated_margin_lines(page_line_lists)
    page_documents: list[Document] = []

    for page_number, (body_raw, tables) in enumerate(
        zip(page_texts, page_tables, strict=True), start=1
    ):
        body_text = _clean_page_text(
            body_raw,
            repeated_margin_lines=repeated_margin_lines,
        )
        blocks: list[str] = []
        if body_text:
            blocks.append(body_text)

        for table in tables:
            table_md = _table_rows_to_markdown(table)
            if table_md:
                blocks.append(table_md)

        if not blocks:
            continue

        page_documents.append(
            Document(
                page_content="\n\n".join(blocks),
                metadata={
                    "source": _policy_source_name(),
                    "page": page_number,
                },
            )
        )

    if not page_documents:
        raise ValueError(f"No extractable content in {policy_path}")

    return page_documents


def _split_into_semantic_units(page_content: str) -> list[str]:
    """
    Split page content into atomic units: paragraphs and tables.

    Double-newline boundaries usually align with paragraph breaks in PDF text
    extraction; Markdown tables are never merged with prose in the same unit.
    """
    units: list[str] = []
    for block in re.split(r"\n{2,}", page_content):
        block = block.strip()
        if not block:
            continue
        if _looks_like_markdown_table(block):
            units.append(block)
            continue
        # Keep natural paragraphs whole; only split long prose later.
        units.append(block)
    return units


def _fallback_split_long_text(text: str, metadata: dict) -> list[Document]:
    """
    Last-resort splitter for units that exceed chunk_size.

    Separators follow sentence and paragraph boundaries before hard cuts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size(),
        chunk_overlap=_chunk_overlap(),
        separators=_SEMANTIC_SEPARATORS,
    )
    return splitter.split_documents(
        [Document(page_content=text, metadata=metadata.copy())]
    )


def _semantic_chunk_page(page_doc: Document) -> list[Document]:
    """Pack paragraph/table units into retrieval chunks without breaking tables."""
    base_meta = {
        "source": page_doc.metadata.get("source", _policy_source_name()),
        "page": page_doc.metadata.get("page"),
    }
    max_size = _chunk_size()
    chunks: list[Document] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        chunks.append(
            Document(
                page_content="\n\n".join(buffer),
                metadata=base_meta.copy(),
            )
        )
        buffer = []
        buffer_len = 0

    for unit in _split_into_semantic_units(page_doc.page_content):
        unit_len = len(unit)
        if unit_len > max_size:
            flush_buffer()
            chunks.extend(_fallback_split_long_text(unit, base_meta))
            continue
        if buffer and buffer_len + 2 + unit_len > max_size:
            flush_buffer()
        buffer.append(unit)
        buffer_len += unit_len + (2 if buffer_len else 0)

    flush_buffer()
    return chunks


def load_and_chunk_policy() -> list[Document]:
    """
    Load the configured policy PDF (e.g. ``compay_policy.pdf``) and produce chunks.

    Pipeline:
    1. pdfplumber per-page text + table extraction
    2. Header/footer/page-number denoising
    3. Structure-aware packing (paragraph / Markdown table atomic units)
    4. RecursiveCharacterTextSplitter fallback for oversized prose only

    Every chunk carries ``metadata['source']`` (filename) and ``metadata['page']``
    (1-based page index) for UI citation and audit trails.
    """
    page_documents = _load_pdf_pages()
    chunks: list[Document] = []
    for page_doc in page_documents:
        chunks.extend(_semantic_chunk_page(page_doc))

    if not chunks:
        raise ValueError("Chunking produced zero documents; check PDF content.")

    return chunks


def build_vectorstore(chunks: list[Document], *, reset_persist: bool = True) -> Chroma:
    """
    Embed chunks with OpenAI-compatible embeddings and persist to local Chroma.
    """
    if reset_persist and _CHROMA_DIR.exists():
        shutil.rmtree(_CHROMA_DIR)

    embeddings = _make_embeddings()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(_CHROMA_DIR),
        collection_name="company_policy",
    )
    _write_chroma_fingerprint()
    return store


def format_docs(docs: list[Document]) -> str:
    """Join retrieved document bodies into a single context string."""
    return "\n\n".join(d.page_content for d in docs)


def _format_docs_for_extraction(docs: list[Document]) -> str:
    """Render retrieval results with source/page metadata for traceable extraction."""
    formatted: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = str(doc.metadata.get("source", _policy_source_name()))
        page = str(doc.metadata.get("page", "N/A"))
        formatted.append(
            f"[Chunk {idx}] source={source}; page={page}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted)


def _extract_json_block(text: str) -> str:
    """Extract JSON body from plain text or fenced code blocks."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1].strip()
    return stripped


def _fallback_parse_audit_report(raw_text: str) -> AuditReport:
    """
    Parse model text into ``AuditReport`` when provider lacks structured output API.
    """
    json_text = _extract_json_block(raw_text)
    try:
        return AuditReport.model_validate_json(json_text)
    except Exception as first_error:
        try:
            # Second chance: parse via json then validate (handles minor JSON artifacts).
            data = json.loads(json_text)
            return AuditReport.model_validate(data)
        except Exception as second_error:
            preview = raw_text.strip().replace("\n", " ")[:300]
            raise ValueError(
                "Failed to parse structured audit output into AuditReport. "
                f"Model output preview: {preview}"
            ) from second_error


def _repair_audit_json_with_llm(
    llm: ChatOpenAI,
    *,
    raw_text: str,
) -> str:
    """
    Ask the model to repair malformed output into strict AuditReport JSON.
    """
    repair_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是一个 JSON 修复器。请将输入内容重写为严格合法的 JSON，"
                    "并严格匹配该 Schema：\n{schema}"
                    "。不要输出任何解释。"
                ),
            ),
            ("human", "待修复内容：\n{raw_text}"),
        ]
    )
    repair_chain = repair_prompt | llm | StrOutputParser()
    return repair_chain.invoke({"raw_text": raw_text, "schema": _AUDIT_REPORT_SCHEMA_JSON})


def reranker_model_name() -> str:
    """Cross-encoder model for reranking; override via RAG_RERANKER_MODEL in .env."""
    return os.getenv("RAG_RERANKER_MODEL", _DEFAULT_RERANKER_MODEL).strip()


def hybrid_recall_k() -> int:
    """Per-leg recall before rerank; override via RAG_RECALL_K (default 10)."""
    raw = os.getenv("RAG_RECALL_K", "10")
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def policy_bm25_cache_key() -> str:
    """Invalidate BM25 cache when the policy PDF changes."""
    path = _policy_file()
    if not path.exists():
        return "missing"
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@lru_cache(maxsize=2)
def _get_cached_cross_encoder(model_name: str) -> HuggingFaceCrossEncoder:
    """Load reranker once per process (avoids reload on every Streamlit rerun)."""
    return HuggingFaceCrossEncoder(model_name=model_name)


@lru_cache(maxsize=2)
def _get_cached_bm25_retriever(policy_key: str) -> BM25Retriever:
    """Build BM25 index once per policy file revision."""
    del policy_key  # only used for cache invalidation
    chunks = load_and_chunk_policy()
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = hybrid_recall_k()
    return retriever


def build_compression_retriever(
    vectorstore: Chroma,
    chunks: list[Document] | None = None,
    *,
    top_n: int = 4,
) -> ContextualCompressionRetriever:
    """
    Hybrid recall (dense + BM25, k each) then cross-encoder rerank to top_n=4.

    Dense leg: Chroma vector search. Sparse leg: BM25 over supplied chunks, or
    the cached demo policy when chunks are omitted. Managed library chunk IDs
    keep equal text from different documents distinct during fusion.
    Both feed an EnsembleRetriever (0.5 / 0.5), then ContextualCompressionRetriever
    applies the configured reranker (default: ms-marco-MiniLM for speed).
    """
    recall_k = hybrid_recall_k()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": recall_k})
    bm25_retriever = (BM25Retriever.from_documents(chunks) if chunks
                      else _get_cached_bm25_retriever(policy_bm25_cache_key()))
    bm25_retriever.k = recall_k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        id_key="chunk_id" if chunks and all("chunk_id" in d.metadata for d in chunks) else None,
    )

    cross_encoder = _get_cached_cross_encoder(reranker_model_name())
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=max(1, top_n))
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )


def build_rag_chain(vectorstore: Chroma, chunks: list[Document]):
    """
    Assemble an LCEL RAG chain: retrieve -> prompt -> LLM -> string output.
    """
    compression_retriever = build_compression_retriever(vectorstore, chunks)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat"),
        temperature=0,
        api_key=_optional_str("OPENAI_API_KEY"),
        base_url=_optional_str("OPENAI_BASE_URL"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRICT_GROUNDED_QA_SYSTEM),
            ("human", "{question}"),
        ]
    )

    # LCEL: parallel keys — same question fans out to retrieval and the chat template.
    rag_chain = (
        {
            "context": itemgetter("question")
            | compression_retriever
            | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def answer_question(
    vectorstore: Chroma,
    chunks: list[Document],
    question: str,
) -> str:
    """Run the RAG chain for a single natural-language question."""
    chain = build_rag_chain(vectorstore, chunks)
    return chain.invoke({"question": question})


def generate_audit_checklist(
    vectorstore: Chroma,
    chunks: list[Document],
    topic: str,
) -> AuditReport:
    """
    Extract structured compliance checklist for a specific policy topic.

    Uses hybrid retrieval + reranking to gather evidence, then invokes a
    structured-output LLM that returns ``AuditReport`` directly.
    """
    compression_retriever = build_compression_retriever(vectorstore, chunks)
    retrieved_docs = compression_retriever.invoke(topic)
    context = _format_docs_for_extraction(retrieved_docs)

    base_llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat"),
        temperature=0,
        api_key=_optional_str("OPENAI_API_KEY"),
        base_url=_optional_str("OPENAI_BASE_URL"),
    )

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是一个企业合规审计员，请基于以下检索到的文档片段，"
                    "提取关于 {topic} 的所有合规红线，并严格按照提供的 JSON "
                    "Schema 格式输出。"
                ),
            ),
            (
                "human",
                "审计主题：{topic}\n\n"
                "文档片段如下：\n{context}\n\n"
                "要求：\n"
                "1) 覆盖检索结果中可支持的所有合规红线；\n"
                '2) amount_limit 无明确金额时必须填 "N/A"；\n'
                '3) exceptions 无明确例外时填 "N/A"；\n'
                "4) source_page 必须引用对应页码。\n"
                "5) 仅输出 JSON 对象，不要输出额外解释。",
            ),
        ]
    )

    # Provider compatibility:
    # OpenAI-compatible gateways (e.g., DeepSeek endpoint) may reject
    # response_format/parse APIs. To guarantee compatibility, we only use
    # JSON-text generation + local Pydantic validation.
    fallback_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "你是一个企业合规审计员。请严格输出一个 JSON 对象，"
                    "且字段必须符合该 Schema：\n{schema}"
                ),
            ),
            (
                "human",
                "审计主题：{topic}\n\n"
                "文档片段如下：\n{context}\n\n"
                "要求：\n"
                "1) 提取关于该主题的所有合规红线；\n"
                '2) amount_limit 无明确金额时填写 "N/A"；\n'
                '3) exceptions 无明确例外时填写 "N/A"；\n'
                "4) source_page 引用具体页码；\n"
                "5) 只返回 JSON，不返回其他文本。",
            ),
        ]
    )
    fallback_chain = fallback_prompt | base_llm | StrOutputParser()
    raw = fallback_chain.invoke(
        {
            "topic": topic,
            "context": context,
            "schema": _AUDIT_REPORT_SCHEMA_JSON,
        }
    )
    try:
        return _fallback_parse_audit_report(raw)
    except ValueError:
        repaired_raw = _repair_audit_json_with_llm(base_llm, raw_text=raw)
        return _fallback_parse_audit_report(repaired_raw)


if __name__ == "__main__":
    _load_env()

    chunks = load_and_chunk_policy()
    vs = build_vectorstore(chunks, reset_persist=True)

    test_question = (
        "According to the Business Conduct Policy, what are the rules on "
        "employees accepting gifts from suppliers?"
    )
    print("Question:", test_question)
    try:
        print("Answer:", answer_question(vs, chunks, test_question))
    except APIStatusError as exc:
        # Typical provider issues: 402 insufficient balance, 401 invalid key, 429 rate limits.
        print(
            "Answer: <LLM request failed - check API key, base URL, and account balance.>",
            f"Provider status: {exc.status_code}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


__all__ = [
    "answer_question",
    "build_compression_retriever",
    "build_rag_chain",
    "build_vectorstore",
    "chroma_index_fingerprint",
    "chroma_index_is_stale",
    "format_docs",
    "generate_audit_checklist",
    "get_embeddings",
    "hybrid_recall_k",
    "load_and_chunk_policy",
    "policy_bm25_cache_key",
    "reranker_model_name",
    "AuditRecord",
    "AuditReport",
]
