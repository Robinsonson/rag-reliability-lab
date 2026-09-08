"""
Automated LLM-as-a-Judge evaluation pipeline for the local RAG system.

Usage:
    python run_eval.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from kb_prompts import STRICT_GROUNDED_QA_SYSTEM
from rag_engine import (
    build_compression_retriever,
    build_vectorstore,
    format_docs,
    load_and_chunk_policy,
)

_BASE_DIR = Path(__file__).resolve().parent
_DATASET_PATH = _BASE_DIR / "eval_dataset.json"
_ALLOWED_SCORES = (0.0, 0.5, 1.0)

_JUDGE_SYSTEM_PROMPT = (
    "你是一个严谨的 RAG 评估裁判。请比较 [标准答案] 和 [RAG生成的答案]。"
    "如果生成的答案与标准答案语义一致且未包含虚假信息，输出 1.0 分；"
    "如果部分一致且无致命事实错误，输出 0.5 分；"
    "如果有幻觉、完全答错或未回答，输出 0.0 分。"
    '你的输出必须是严格的 JSON 格式：{{"score": 1.0, "reason": "简短的理由"}}。'
)


class RetrieverLike(Protocol):
    def invoke(self, query: str) -> Any: ...


def _optional_str(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _load_env() -> None:
    load_dotenv(_BASE_DIR / ".env")


def _load_eval_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{idx} is not a JSON object.")
        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        if not question or not ground_truth:
            raise ValueError(
                f"Dataset item #{idx} must include non-empty question and ground_truth."
            )
        normalized.append({"question": question, "ground_truth": ground_truth})
    return normalized


def _extract_json_like(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        return text[first : last + 1].strip()
    return stripped


def _normalize_score(raw_score: Any) -> float:
    if isinstance(raw_score, str):
        score = float(raw_score.strip())
    else:
        score = float(raw_score)
    nearest = min(_ALLOWED_SCORES, key=lambda x: abs(x - score))
    return float(nearest)


def _parse_judge_output(raw_text: str) -> tuple[float, str]:
    """
    Parse model output into (score, reason) with robust fallbacks.
    """
    try:
        parsed = json.loads(_extract_json_like(raw_text))
        score = _normalize_score(parsed.get("score", 0.0))
        reason = str(parsed.get("reason", "No reason provided.")).strip()
        return score, reason or "No reason provided."
    except Exception:
        score_match = re.search(r"\b(1(?:\.0)?|0\.5|0(?:\.0)?)\b", raw_text)
        score = _normalize_score(score_match.group(1) if score_match else 0.0)

        reason_match = re.search(
            r"(?:reason|理由)\s*[:：]\s*(.+)",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            compact = " ".join(raw_text.split())
            reason = compact[:180] if compact else "Judge output could not be parsed."
        return score, reason


def _build_judge_chain() -> Any:
    judge_llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat"),
        temperature=0,
        api_key=_optional_str("OPENAI_API_KEY"),
        base_url=_optional_str("OPENAI_BASE_URL"),
    )
    judge_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _JUDGE_SYSTEM_PROMPT),
            (
                "human",
                "[问题]\n{question}\n\n"
                "[标准答案]\n{ground_truth}\n\n"
                "[RAG生成的答案]\n{prediction}\n\n"
                "请仅输出严格 JSON。",
            ),
        ]
    )
    return judge_prompt | judge_llm | StrOutputParser()


def _build_answer_chain() -> Any:
    answer_llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat"),
        temperature=0,
        api_key=_optional_str("OPENAI_API_KEY"),
        base_url=_optional_str("OPENAI_BASE_URL"),
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRICT_GROUNDED_QA_SYSTEM),
            ("human", "{question}"),
        ]
    )
    return qa_prompt | answer_llm | StrOutputParser()


def _print_block(title: str, content: str) -> None:
    print(f"\n{title}")
    print(content if content.strip() else "<empty>")


def _safe_generate_answer(
    *,
    retriever: RetrieverLike,
    answer_chain: Any,
    question: str,
) -> str:
    docs = retriever.invoke(question)
    if not isinstance(docs, list):
        return "<RAG generation failed: retriever returned non-list context>"
    context = format_docs(docs)
    return str(answer_chain.invoke({"question": question, "context": context})).strip()


def evaluate_pipeline(
    retriever_name: str,
    retriever: RetrieverLike,
    dataset: list[dict[str, str]],
    *,
    answer_chain: Any,
    judge_chain: Any,
) -> float:
    scores: list[float] = []
    total = len(dataset)
    print("\n" + "=" * 80)
    print(f"PIPELINE: {retriever_name}".center(80))
    print("=" * 80)

    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print("\n" + "-" * 80)
        print(f"[{retriever_name}] Case {idx:02d}/{total}")

        try:
            prediction = _safe_generate_answer(
                retriever=retriever,
                answer_chain=answer_chain,
                question=question,
            )
        except Exception as exc:
            prediction = f"<RAG generation failed: {exc}>"

        try:
            judge_raw = judge_chain.invoke(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                }
            )
            score, reason = _parse_judge_output(judge_raw)
        except Exception as exc:
            score = 0.0
            reason = f"Judge failed: {exc}"

        scores.append(score)
        _print_block("Question:", question)
        _print_block("Generated Answer:", prediction)
        _print_block("Score:", str(score))
        _print_block("Reason:", reason)

    avg_score = mean(scores) if scores else 0.0
    print("\n" + "-" * 80)
    print(f"[{retriever_name}] Average Score (Accuracy): {avg_score * 100:.2f}%")
    return avg_score


def _print_ablation_report(baseline_avg: float, advanced_avg: float) -> None:
    improvement = advanced_avg - baseline_avg
    rows = [
        ("Baseline Average Score", f"{baseline_avg * 100:.2f}%"),
        ("Advanced Average Score", f"{advanced_avg * 100:.2f}%"),
        ("提升幅度 (Improvement)", f"{improvement * 100:+.2f}%"),
    ]
    key_width = max(len(k) for k, _ in rows)
    val_width = max(len(v) for _, v in rows)
    border = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"

    print("\n" + "=" * 80)
    print("ABLATION STUDY REPORT".center(80))
    print("=" * 80)
    print(border)
    for key, value in rows:
        print(f"| {key.ljust(key_width)} | {value.rjust(val_width)} |")
    print(border)


def run_eval() -> int:
    _load_env()
    if not _optional_str("OPENAI_API_KEY"):
        print("ERROR: Missing OPENAI_API_KEY in .env", file=sys.stderr)
        return 1

    print("=" * 80)
    print("RAG EVALUATION (LLM-as-a-Judge)".center(80))
    print("=" * 80)
    print("Loading policy and building vector store...")

    try:
        chunks = load_and_chunk_policy()
        vectorstore = build_vectorstore(chunks, reset_persist=False)
        dataset = _load_eval_dataset(_DATASET_PATH)
        answer_chain = _build_answer_chain()
        judge_chain = _build_judge_chain()
    except Exception as exc:
        print(f"Initialization failed: {exc}", file=sys.stderr)
        return 1

    total = len(dataset)
    print(f"Loaded {total} evaluation samples.")

    baseline_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    advanced_retriever = build_compression_retriever(vectorstore, chunks)

    baseline_avg = evaluate_pipeline(
        "Baseline (Dense k=4)",
        baseline_retriever,
        dataset,
        answer_chain=answer_chain,
        judge_chain=judge_chain,
    )
    advanced_avg = evaluate_pipeline(
        "Advanced (Hybrid + Rerank)",
        advanced_retriever,
        dataset,
        answer_chain=answer_chain,
        judge_chain=judge_chain,
    )
    _print_ablation_report(baseline_avg, advanced_avg)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_eval())
