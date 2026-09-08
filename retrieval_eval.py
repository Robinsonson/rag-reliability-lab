"""Deterministic evidence-retrieval evaluation for the policy corpus.

This module deliberately does not call a chat model. A case passes only when a
human-labelled evidence page appears in the ranked retrieval results.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma

from rag_engine import (
    build_compression_retriever,
    build_vectorstore,
    chroma_index_is_stale,
    get_embeddings,
    load_and_chunk_policy,
    reranker_model_name,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
TEST_DATASET_PATH = BASE_DIR / "eval_dataset_test.json"
MANIFEST_PATH = BASE_DIR / "evaluation_manifest.json"
REPORT_DIR = BASE_DIR / "evaluation_runs"
LATEST_REPORT = REPORT_DIR / "latest.json"


def load_retrieval_cases(
    path: Path = DATASET_PATH,
    *,
    split: str = "development",
) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    cases: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{index} is not an object.")
        question = str(item.get("question", "")).strip()
        pages = item.get("evidence_pages")
        if not question or not isinstance(pages, list) or not pages:
            raise ValueError(
                f"Dataset item #{index} needs a question and evidence_pages."
            )
        normalized_pages = sorted({int(page) for page in pages if int(page) > 0})
        if not normalized_pages:
            raise ValueError(f"Dataset item #{index} has no valid evidence page.")
        cases.append(
            {
                "id": index,
                "case_id": f"{split}-{index:02d}",
                "split": split,
                "question": question,
                "evidence_pages": normalized_pages,
                "evidence_section": str(item.get("evidence_section", "")),
                "label_status": str(item.get("label_status", "human_verified")),
            }
        )
    return cases


def load_all_retrieval_cases() -> list[dict[str, Any]]:
    return load_retrieval_cases(DATASET_PATH, split="development") + load_retrieval_cases(
        TEST_DATASET_PATH,
        split="test",
    )


def page_ranking(docs: list[Any]) -> list[int]:
    return [
        int(doc.metadata["page"])
        for doc in docs
        if getattr(doc, "metadata", {}).get("page") is not None
    ]


def first_relevant_rank(ranked_pages: list[int], evidence_pages: list[int]) -> int | None:
    expected = set(evidence_pages)
    return next(
        (rank for rank, page in enumerate(ranked_pages, start=1) if page in expected),
        None,
    )


def summarize_pipeline(cases: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    ranks = [case["first_relevant_rank"] for case in cases]
    latencies = [float(case["latency_ms"]) for case in cases]
    hit_count = sum(rank is not None and rank <= k for rank in ranks)
    reciprocal_ranks = [
        1.0 / rank if rank is not None and rank <= k else 0.0 for rank in ranks
    ]
    return {
        "evidence_page_hit_rate_at_k": (
            round(hit_count / len(cases), 4) if cases else 0.0
        ),
        "mrr_at_k": round(mean(reciprocal_ranks), 4) if cases else 0.0,
        "hit_count": hit_count,
        "miss_count": len(cases) - hit_count,
        "case_count": len(cases),
        "median_latency_ms": round(median(latencies)) if latencies else 0,
        "mean_latency_ms": round(mean(latencies)) if latencies else 0,
    }


def _load_vectorstore(chunks: list[Any]) -> Chroma:
    if chroma_index_is_stale():
        return build_vectorstore(chunks, reset_persist=True)
    return Chroma(
        persist_directory=str(BASE_DIR / "chroma_db"),
        embedding_function=get_embeddings(),
        collection_name="company_policy",
    )


def evaluate_retrieval(*, k: int = 4, save: bool = True) -> dict[str, Any]:
    if k < 1:
        raise ValueError("k must be at least 1.")

    load_dotenv(BASE_DIR / ".env")
    cases = load_all_retrieval_cases()
    chunks = load_and_chunk_policy()
    vectorstore = _load_vectorstore(chunks)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    advanced_retriever = build_compression_retriever(
        vectorstore,
        chunks,
        top_n=k,
    )
    pipeline_cases: dict[str, list[dict[str, Any]]] = {
        "dense": [],
        "hybrid_rerank": [],
    }
    for case in cases:
        started = perf_counter()
        dense_docs = dense_retriever.invoke(case["question"])
        dense_latency_ms = round((perf_counter() - started) * 1000)
        dense_pages = page_ranking(dense_docs)
        pipeline_cases["dense"].append(
            {
                **case,
                "retrieved_pages": dense_pages,
                "first_relevant_rank": first_relevant_rank(
                    dense_pages,
                    case["evidence_pages"],
                ),
                "latency_ms": dense_latency_ms,
            }
        )

        started = perf_counter()
        candidate_docs = advanced_retriever.base_retriever.invoke(case["question"])
        candidate_pages = page_ranking(candidate_docs)
        reranked_docs = list(
            advanced_retriever.base_compressor.compress_documents(
                candidate_docs,
                case["question"],
            )
        )
        advanced_latency_ms = round((perf_counter() - started) * 1000)
        advanced_pages = page_ranking(reranked_docs)
        advanced_rank = first_relevant_rank(
            advanced_pages,
            case["evidence_pages"],
        )
        candidate_rank = first_relevant_rank(
            candidate_pages,
            case["evidence_pages"],
        )
        pipeline_cases["hybrid_rerank"].append(
            {
                **case,
                "candidate_pages": candidate_pages,
                "retrieved_pages": advanced_pages,
                "first_relevant_rank": advanced_rank,
                "failure_stage": (
                    None
                    if advanced_rank is not None
                    else "reranking"
                    if candidate_rank is not None
                    else "recall"
                ),
                "latency_ms": advanced_latency_ms,
            }
        )

    report = {
        "schema_version": 3,
        "metric_name": f"Evidence-page Hit Rate@{k}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_definition": (
            "A hit occurs when at least one human-labelled evidence page appears "
            f"in the first {k} retrieved chunks."
        ),
        "configuration": {
            "k": k,
            "embedding_backend": os.getenv("RAG_EMBEDDING_BACKEND", "openai"),
            "embedding_model": os.getenv("HF_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "default",
            "reranker_model": reranker_model_name(),
        },
        "dataset": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
        "pipelines": [],
    }
    names = {
        "dense": "Dense baseline",
        "hybrid_rerank": "Hybrid + rerank",
    }
    for pipeline_id, results in pipeline_cases.items():
        report["pipelines"].append(
            {
                "id": pipeline_id,
                "name": names[pipeline_id],
                "metrics": summarize_pipeline(results, k=k),
                "metrics_by_split": {
                    split: summarize_pipeline(
                        [case for case in results if case["split"] == split],
                        k=k,
                    )
                    for split in ("development", "test")
                },
                "cases": results,
            }
        )

    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def load_latest_report() -> dict[str, Any] | None:
    if not LATEST_REPORT.exists():
        return None
    return json.loads(LATEST_REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    result = evaluate_retrieval()
    k = result["configuration"]["k"]
    for pipeline in result["pipelines"]:
        metrics = pipeline["metrics"]
        print(
            f"{pipeline['name']} (all): Evidence-page Hit Rate@{k}="
            f"{metrics['evidence_page_hit_rate_at_k']:.1%}, "
            f"MRR@{k}={metrics['mrr_at_k']:.3f}"
        )
        for split, split_metrics in pipeline["metrics_by_split"].items():
            print(
                f"  {split}: n={split_metrics['case_count']}, "
                f"Evidence-page Hit Rate@{k}="
                f"{split_metrics['evidence_page_hit_rate_at_k']:.1%}, "
                f"MRR@{k}={split_metrics['mrr_at_k']:.3f}, "
                f"median={split_metrics['median_latency_ms']} ms"
            )
