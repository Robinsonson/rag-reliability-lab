"""FastAPI backend for the RAG Reliability Lab."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
EVAL_DATASET = BASE_DIR / "eval_dataset.json"
TEST_EVAL_DATASET = BASE_DIR / "eval_dataset_test.json"
POLICY_CANDIDATES = (BASE_DIR / "company_policy.pdf", BASE_DIR / "compay_policy.pdf")

# Keep model downloads project-local by default. This avoids permission and
# portability failures caused by machine-level Hugging Face cache locations.
MODEL_CACHE_DIR = BASE_DIR / ".model_cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR))

app = FastAPI(title="RAG Reliability Lab", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    expected_pages: list[int] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def _policy_path() -> Path:
    configured = (os.getenv("RAG_POLICY_FILE") or "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else BASE_DIR / path
    return next((path for path in POLICY_CANDIDATES if path.exists()), POLICY_CANDIDATES[0])


def _eval_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in (EVAL_DATASET, TEST_EVAL_DATASET):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            cases.extend(data)
    return cases


def _document_payload(doc: Any, rank: int) -> dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return {
        "rank": rank,
        "page": metadata.get("page"),
        "source": metadata.get("source", "unknown"),
        "content": str(getattr(doc, "page_content", "")).strip(),
    }


@lru_cache(maxsize=2)
def _cached_runtime(index_fingerprint: str) -> tuple[Any, list[Any], Any, Any]:
    del index_fingerprint
    from langchain_chroma import Chroma

    from rag_engine import (
        build_compression_retriever,
        build_vectorstore,
        chroma_index_is_stale,
        get_embeddings,
        load_and_chunk_policy,
    )

    chunks = load_and_chunk_policy()
    if chroma_index_is_stale():
        vectorstore = build_vectorstore(chunks, reset_persist=True)
    else:
        vectorstore = Chroma(
            persist_directory=str(BASE_DIR / "chroma_db"),
            embedding_function=get_embeddings(),
            collection_name="company_policy",
        )
    baseline = vectorstore.as_retriever(search_kwargs={"k": 4})
    advanced = build_compression_retriever(vectorstore, chunks)
    return vectorstore, chunks, baseline, advanced


def _runtime() -> tuple[Any, list[Any], Any, Any]:
    load_dotenv(BASE_DIR / ".env")
    from rag_engine import chroma_index_fingerprint

    return _cached_runtime(chroma_index_fingerprint())


def _diagnose(
    baseline_docs: list[Any],
    advanced_docs: list[Any],
    expected_pages: list[int],
    hybrid_docs: list[Any] | None = None,
) -> dict[str, str]:
    if not expected_pages:
        return {
            "stage": "unlabelled",
            "summary": "Add an expected page to turn this trace into a labelled diagnostic.",
        }

    expected = set(expected_pages)
    baseline_pages = {doc.metadata.get("page") for doc in baseline_docs}
    hybrid_pages = {
        doc.metadata.get("page") for doc in (hybrid_docs or [])
    }
    advanced_pages = {doc.metadata.get("page") for doc in advanced_docs}
    baseline_hit = bool(expected & baseline_pages)
    hybrid_hit = bool(expected & hybrid_pages)
    advanced_hit = bool(expected & advanced_pages)

    if advanced_hit:
        return {
            "stage": "passed",
            "summary": "The advanced pipeline retained at least one expected evidence page.",
        }
    if hybrid_hit:
        return {
            "stage": "reranking",
            "summary": "Hybrid recall found the evidence, but the cross-encoder removed it.",
        }
    if baseline_hit:
        return {
            "stage": "fusion",
            "summary": "Dense top-k found the evidence, but hybrid fusion lost the candidate.",
        }
    return {
        "stage": "recall",
        "summary": "Neither pipeline recalled an expected evidence page in the returned set.",
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    policy = _policy_path()
    cases = _eval_cases()
    index_dir = BASE_DIR / "chroma_db"
    return {
        "corpus": {
            "name": policy.name,
            "size_mb": round(policy.stat().st_size / 1_048_576, 2) if policy.exists() else None,
            "available": policy.exists(),
            "index_present": index_dir.exists() and any(index_dir.iterdir()),
        },
        "evaluation": {
            "case_count": len(cases),
            "labelled_evidence_count": sum(1 for item in cases if item.get("evidence_pages")),
            "latest_run_available": (BASE_DIR / "evaluation_runs" / "latest.json").exists(),
        },
        "pipelines": [
            {"id": "dense", "name": "Dense baseline", "detail": "Vector search · top 4"},
            {
                "id": "hybrid_rerank",
                "name": "Hybrid + rerank",
                "detail": "Dense + BM25 · cross-encoder · top 4",
            },
        ],
        "configuration": {
            "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "900")),
            "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "80")),
            "recall_k": int(os.getenv("RAG_RECALL_K", "10")),
            "reranker": os.getenv(
                "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
            ),
        },
    }


@app.get("/api/evaluations/latest")
def latest_evaluation() -> dict[str, Any]:
    from retrieval_eval import load_latest_report

    report = load_latest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No saved retrieval evaluation yet.")
    return report


@app.post("/api/evaluations/run-retrieval")
def run_retrieval_evaluation() -> dict[str, Any]:
    try:
        from retrieval_eval import evaluate_retrieval

        return evaluate_retrieval(k=4, save=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"The retrieval evaluation could not complete. Detail: {exc}",
        ) from exc


@app.post("/api/retrieval/compare")
def compare_retrieval(request: RetrievalRequest) -> dict[str, Any]:
    try:
        _, _, baseline, advanced = _runtime()

        started = perf_counter()
        baseline_docs = baseline.invoke(request.query)
        baseline_ms = round((perf_counter() - started) * 1000)

        started = perf_counter()
        hybrid_docs = advanced.base_retriever.invoke(request.query)
        hybrid_ms = round((perf_counter() - started) * 1000)

        started = perf_counter()
        advanced_docs = list(
            advanced.base_compressor.compress_documents(hybrid_docs, request.query)
        )
        rerank_ms = round((perf_counter() - started) * 1000)
        advanced_ms = hybrid_ms + rerank_ms
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                "The retrieval pipeline could not start. Check the embedding and "
                f"reranker configuration. Detail: {exc}"
            ),
        ) from exc

    return {
        "query": request.query,
        "expected_pages": request.expected_pages,
        "diagnosis": _diagnose(
            baseline_docs,
            advanced_docs,
            request.expected_pages,
            hybrid_docs,
        ),
        "pipelines": [
            {
                "id": "dense",
                "name": "Dense baseline",
                "latency_ms": baseline_ms,
                "results": [
                    _document_payload(doc, rank)
                    for rank, doc in enumerate(baseline_docs, start=1)
                ],
            },
            {
                "id": "hybrid_recall",
                "name": "Hybrid candidates",
                "latency_ms": hybrid_ms,
                "results": [
                    _document_payload(doc, rank)
                    for rank, doc in enumerate(hybrid_docs, start=1)
                ],
            },
            {
                "id": "hybrid_rerank",
                "name": "Hybrid + rerank",
                "latency_ms": advanced_ms,
                "rerank_latency_ms": rerank_ms,
                "results": [
                    _document_payload(doc, rank)
                    for rank, doc in enumerate(advanced_docs, start=1)
                ],
            },
        ],
    }


@app.post("/api/answer")
def answer_question(request: AnswerRequest) -> dict[str, Any]:
    try:
        from grounded_answer import generate_grounded_answer

        _, _, _, advanced = _runtime()
        started = perf_counter()
        docs = advanced.invoke(request.question)
        retrieval_ms = round((perf_counter() - started) * 1000)
        started = perf_counter()
        result = generate_grounded_answer(request.question, docs)
        generation_ms = round((perf_counter() - started) * 1000)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"The grounded answer pipeline could not complete. Detail: {exc}",
        ) from exc

    return {
        "question": request.question,
        **result,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "retrieved": [
            _document_payload(doc, rank) for rank, doc in enumerate(docs, start=1)
        ],
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lab_api:app", host="127.0.0.1", port=8000, reload=False)
