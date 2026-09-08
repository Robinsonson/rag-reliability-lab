"""Grounded answer contract with validated passage citations and abstention."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


REFUSAL_MESSAGE = "I cannot answer this from the available policy evidence."


def _optional_str(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def format_evidence(docs: list[Document]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = str(doc.metadata.get("source", "unknown"))
        page = str(doc.metadata.get("page", "unknown"))
        blocks.append(
            f"[S{index}] source={source}; page={page}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    elif not (stripped.startswith("{") and stripped.endswith("}")):
        first, last = stripped.find("{"), stripped.rfind("}")
        if first >= 0 and last > first:
            stripped = stripped[first : last + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("The answer model did not return a JSON object.")
    return parsed


def validate_answer_payload(
    payload: dict[str, Any],
    docs: list[Document],
) -> dict[str, Any]:
    """Map model-selected passage IDs to server-owned source metadata."""
    raw_answerable = payload.get("answerable")
    answerable = raw_answerable is True or (
        isinstance(raw_answerable, str) and raw_answerable.strip().lower() == "true"
    )
    answer = str(payload.get("answer", "")).strip()
    raw_ids = payload.get("citation_ids", [])
    citation_ids = raw_ids if isinstance(raw_ids, list) else []

    valid: list[tuple[str, Document]] = []
    seen: set[str] = set()
    for raw_id in citation_ids:
        passage_id = str(raw_id).strip().upper()
        match = re.fullmatch(r"S([1-9]\d*)", passage_id)
        if not match or passage_id in seen:
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < len(docs):
            valid.append((passage_id, docs[index]))
            seen.add(passage_id)

    if not answerable or not answer or not valid:
        if not answerable:
            validation = "model_abstained"
        elif not answer:
            validation = "empty_answer"
        else:
            validation = "missing_valid_citation"
        return {
            "answerable": False,
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "validation": validation,
        }

    citations = []
    for passage_id, doc in valid:
        content = doc.page_content.strip()
        citations.append(
            {
                "passage_id": passage_id,
                "source": str(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page"),
                "evidence": content[:700],
            }
        )
    return {
        "answerable": True,
        "answer": answer,
        "citations": citations,
        "validation": "citation_ids_validated",
    }


def generate_grounded_answer(
    question: str,
    docs: list[Document],
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    if not docs:
        return validate_answer_payload({"answerable": False}, docs)

    load_dotenv()
    model = llm or ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "deepseek-chat"),
        temperature=0,
        api_key=_optional_str("OPENAI_API_KEY"),
        base_url=_optional_str("OPENAI_BASE_URL"),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer enterprise-policy questions using only the numbered evidence "
                "passages. Set answerable=true when the passages contain the relevant rule, "
                "even when the user's wording differs or the answer requires combining a "
                "general rule with a specific prohibition or exception. A specific rule "
                "overrides a general allowance. Set answerable=false only when none of the "
                "passages contains information that can answer the question. When "
                "answerable=true, cite every passage needed for the answer. Return only "
                "JSON with this shape: {{\"answerable\": true, \"answer\": \"...\", "
                "\"citation_ids\": [\"S1\"]}}. Never invent a passage ID or page number.",
            ),
            ("human", "Question:\n{question}\n\nEvidence:\n{evidence}"),
        ]
    )
    raw = (prompt | model | StrOutputParser()).invoke(
        {"question": question, "evidence": format_evidence(docs)}
    )
    return validate_answer_payload(_extract_json_object(str(raw)), docs)


__all__ = [
    "REFUSAL_MESSAGE",
    "format_evidence",
    "generate_grounded_answer",
    "validate_answer_payload",
]
