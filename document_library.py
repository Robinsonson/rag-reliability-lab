"""Local versioned document library. Immutable originals; one active revision.

The demo corpus/evaluation stays separate. Changes invalidate the library index;
queries never fall back to an old index if rebuilding the active set fails.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024


def parse_pages(name, data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose a nonempty file up to 5 MB")
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        pages = [data.decode("utf-8-sig")]
    elif suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise ValueError("File is not a PDF")
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > 100:
                raise ValueError("PDF limit is 100 pages")
            pages = [page.extract_text() or "" for page in pdf.pages]
    else:
        raise ValueError("Only text PDFs and UTF-8 .txt files are supported")
    if not any(p.strip() for p in pages):
        raise ValueError("No readable text. Scanned documents need OCR, which this library does not provide.")
    if sum(map(len, pages)) > 500_000:
        raise ValueError("Extracted text exceeds 500,000 characters")
    return pages


class DocumentLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "library.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, logical_id TEXT NOT NULL, version INTEGER NOT NULL,
            name TEXT NOT NULL, sha256 TEXT NOT NULL, original BLOB NOT NULL,
            pages TEXT NOT NULL, active INTEGER NOT NULL, created_at TEXT NOT NULL)""")
        self.db.commit()
        self._runtime = None
        self._key = None
        self.last_error = None

    def close(self):
        self.db.close()

    def _rows(self):
        return self.db.execute("SELECT id,logical_id,version,name,sha256,pages,active,created_at FROM documents ORDER BY created_at DESC").fetchall()

    def catalog(self):
        with self.lock:
            rows = self._rows()
            counts = Counter(d.metadata["document_id"] for d in self._runtime[1]) if self._runtime else {}
            return {"documents": [{"id": r["id"], "logical_id": r["logical_id"], "version": r["version"],
                                   "name": r["name"], "sha256": r["sha256"], "active": bool(r["active"]),
                                   "page_count": len(json.loads(r["pages"])), "created_at": r["created_at"],
                                   "chunk_count": counts.get(r["id"], 0) if self._runtime and r["active"] else None}
                                  for r in rows],
                    "index_status": "empty" if not any(r["active"] for r in rows) else
                                    "failed" if self.last_error else "ready" if self._runtime else "pending",
                    "last_error": self.last_error,
                    "chunk_count": len(self._runtime[1]) if self._runtime else None}

    def upload(self, name, encoded, replaces=None):
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise ValueError("A filename up to 160 characters is required")
        # Original names are labels only, never filesystem paths.
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        if not isinstance(encoded, str) or len(encoded) > (MAX_BYTES * 4 // 3 + 4):
            raise ValueError("File exceeds 5 MB")
        data = base64.b64decode(encoded, validate=True)
        pages = parse_pages(name, data)
        digest = hashlib.sha256(data).hexdigest()
        with self.lock, self.db:
            if len(self._rows()) >= 100:
                raise ValueError("Local library limit is 100 retained revisions")
            old = self.db.execute("SELECT * FROM documents WHERE id=?", (replaces,)).fetchone() if replaces else None
            if replaces and not old:
                raise KeyError(replaces)
            if old:
                current = self.db.execute("SELECT * FROM documents WHERE logical_id=? ORDER BY version DESC LIMIT 1", (old["logical_id"],)).fetchone()
                if current["id"] != replaces:
                    raise ValueError("This document has a newer revision. Refresh before replacing it.")
                if current["sha256"] == digest:
                    raise ValueError("The replacement is identical to the current revision")
            elif self.db.execute("SELECT 1 FROM documents WHERE sha256=? AND active=1", (digest,)).fetchone():
                raise ValueError("An identical document is already active")
            identifier = uuid.uuid4().hex
            logical = old["logical_id"] if old else identifier
            version = old["version"] + 1 if old else 1
            self.db.execute("UPDATE documents SET active=0 WHERE logical_id=?", (logical,))
            self.db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                            (identifier, logical, version, name, digest, data, json.dumps(pages), 1,
                             datetime.now(timezone.utc).isoformat()))
            self._invalidate()
            return {"id": identifier, "version": version, "name": name}

    def _invalidate(self):
        if self._runtime:
            self._runtime[0].delete_collection()
        self._runtime = None
        self._key = None
        self.last_error = None

    def activate(self, identifier, active):
        if type(active) is not bool:
            raise ValueError("active must be a boolean")
        with self.lock, self.db:
            row = self.db.execute("SELECT * FROM documents WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise KeyError(identifier)
            if active:
                self.db.execute("UPDATE documents SET active=0 WHERE logical_id=?", (row["logical_id"],))
            self.db.execute("UPDATE documents SET active=? WHERE id=?", (int(active), identifier))
            self._invalidate()

    def original(self, identifier):
        with self.lock:
            row = self.db.execute("SELECT name,original FROM documents WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise KeyError(identifier)
            return row["name"], bytes(row["original"])

    def runtime(self):
        with self.lock:
            from rag_engine import _semantic_chunk_page, _chunk_size, _chunk_overlap, get_embeddings, build_compression_retriever
            from langchain_core.documents import Document
            from langchain_chroma import Chroma
            import os
            rows = [r for r in self._rows() if r["active"]]
            if not rows:
                raise ValueError("The library has no active documents. Upload or activate a document first.")
            configuration = [sorted(r["id"] for r in rows), _chunk_size(), _chunk_overlap(),
                             os.getenv("RAG_EMBEDDING_BACKEND"), os.getenv("HF_EMBEDDING_MODEL"),
                             os.getenv("OPENAI_EMBEDDING_MODEL"), os.getenv("RAG_RERANKER_MODEL"), os.getenv("RAG_RECALL_K")]
            key = hashlib.sha256(json.dumps(configuration).encode()).hexdigest()
            if self._runtime and key == self._key:
                return self._runtime
            self._invalidate()
            chunks = []
            vector = None
            try:
                for row in rows:
                    for page, text in enumerate(json.loads(row["pages"]), 1):
                        parts = _semantic_chunk_page(Document(page_content=text, metadata={"source": row["name"], "page": page}))
                        for part in parts:
                            part.metadata.update(document_id=row["id"], version=row["version"],
                                                 chunk_id=f"{row['id']}:{len(chunks)}")
                            chunks.append(part)
                            if len(chunks) > 5000:
                                raise ValueError("Active library exceeds 5,000 chunks; deactivate documents first")
                # A private generation prevents partial rebuilds from serving queries.
                vector = Chroma(collection_name="library_" + uuid.uuid4().hex, embedding_function=get_embeddings())
                vector.add_documents(chunks, ids=[d.metadata["chunk_id"] for d in chunks])
                advanced = build_compression_retriever(vector, chunks)
                runtime = (vector, chunks, vector.as_retriever(search_kwargs={"k": 4}), advanced)
                self._runtime, self._key, self.last_error = runtime, key, None
                return runtime
            except Exception:
                if vector is not None:
                    vector.delete_collection()
                self._runtime = None
                self.last_error = "Index build failed. Check model availability and retry; no old index is served."
                raise
