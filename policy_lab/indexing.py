"""Bounded chunking and model adapters shared by ingestion and retrieval."""
from __future__ import annotations

import hashlib
import math
import re


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lexical_tokens(text: str) -> list[str]:
    """Unicode word tokens plus CJK unigrams/bigrams for PostgreSQL simple FTS."""
    tokens = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower())
    result = []
    for token in tokens:
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            result.extend(token)
            result.extend(token[i : i + 2] for i in range(len(token) - 1))
        else:
            result.append(token)
    return result


def make_chunks(title, blocks, size=900, overlap=100):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk settings")
    for block in blocks:
        raw = block["text"]
        for start in range(0, len(raw), size - overlap):
            text = raw[start : start + size]
            context = f"{title}\n{block.get('heading', '')}\n{text}"
            yield {"block_id": block["id"], "text": text, "context": context,
                   "lexical": " ".join(lexical_tokens(context)), "start": start}
            if start + size >= len(raw):
                break


class SentenceEmbedder:
    """Models must be explicitly provisioned; request handling never downloads models."""
    def __init__(self, settings):
        self.settings = settings
        self._model = None

    def encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.settings.embedding_model,
                cache_folder=str(self.settings.model_cache_dir), local_files_only=True)
        return self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False).tolist()


class Reranker:
    def __init__(self, settings):
        self.settings = settings
        self._model = None

    def rank(self, query, evidence):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.settings.reranker_model,
                cache_folder=str(self.settings.model_cache_dir), local_files_only=True)
        scores = self._model.predict([(query, row["text"]) for row in evidence])
        return sorted(zip(evidence, map(float, scores)), key=lambda item: item[1], reverse=True)


def cosine(a, b):
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / denominator if denominator else 0.0
