import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
import retrieval_eval as evaluation


class AblationTests(unittest.TestCase):
    def test_fusion_and_reranking_are_scored_separately_and_runs_are_retained(self):
        docs = [Document(page_content=str(p), metadata={"page": p}) for p in (1, 2, 3)]
        cases = [{"case_id": "test-01", "split": "test", "question": "q", "evidence_pages": [3]}]
        dense = Mock()
        dense.invoke.return_value = docs[:2]
        sparse = Mock()
        sparse.invoke.return_value = [docs[2], docs[0]]
        hybrid = Mock()
        hybrid.invoke.return_value = docs
        compressor = Mock()
        compressor.compress_documents.return_value = [docs[2], docs[0]]
        advanced = SimpleNamespace(base_retriever=hybrid, base_compressor=compressor)
        vector = Mock()
        vector.as_retriever.return_value = dense
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (patch.object(evaluation, "load_all_retrieval_cases", return_value=cases),
                  patch.object(evaluation, "load_and_chunk_policy", return_value=docs),
                  patch.object(evaluation, "_load_vectorstore", return_value=vector),
                  patch.object(evaluation, "build_compression_retriever", return_value=advanced),
                  patch.object(evaluation.BM25Retriever, "from_documents", return_value=sparse),
                  patch.object(evaluation, "chroma_index_fingerprint", return_value="fixture"),
                  patch.object(evaluation, "REPORT_DIR", root),
                  patch.object(evaluation, "LATEST_REPORT", root / "latest.json")):
                first = evaluation.evaluate_retrieval(k=2, warmup=False)
                second = evaluation.evaluate_retrieval(k=2, warmup=False)
            pipelines = {p["id"]: p for p in first["pipelines"]}
            self.assertEqual(first["schema_version"], 4)
            self.assertEqual(pipelines["hybrid"]["metrics"]["hit_count"], 0)
            self.assertEqual(pipelines["hybrid_rerank"]["metrics"]["hit_count"], 1)
            self.assertEqual(pipelines["bm25"]["metrics"]["mrr_at_k"], 1)
            self.assertEqual(pipelines["dense"]["metrics"]["hit_count"], 0)
            self.assertEqual(pipelines["hybrid"]["cases"][0]["retrieved_pages"], [1, 2])
            self.assertEqual(len(list(root.glob("run-*.json"))), 2)
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual(json.loads((root / "latest.json").read_text())["run_id"], second["run_id"])
            self.assertEqual(pipelines["bm25"]["metrics_by_split"]["test"]["case_count"], 1)
