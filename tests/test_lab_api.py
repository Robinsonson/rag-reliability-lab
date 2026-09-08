from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from grounded_answer import REFUSAL_MESSAGE, validate_answer_payload
from lab_api import _diagnose, app
from retrieval_eval import (
    first_relevant_rank,
    load_all_retrieval_cases,
    summarize_pipeline,
)


BASE_DIR = Path(__file__).resolve().parents[1]


class FakeDocument:
    def __init__(self, page: int):
        self.metadata = {"page": page}


class DiagnosisTests(unittest.TestCase):
    def test_unlabelled_trace_does_not_claim_a_failure(self) -> None:
        result = _diagnose([FakeDocument(1)], [FakeDocument(2)], [])
        self.assertEqual(result["stage"], "unlabelled")

    def test_expected_page_retained_by_advanced_pipeline_passes(self) -> None:
        result = _diagnose([FakeDocument(1)], [FakeDocument(7)], [7])
        self.assertEqual(result["stage"], "passed")

    def test_dense_hit_removed_by_advanced_pipeline_is_reranking_failure(self) -> None:
        result = _diagnose(
            [FakeDocument(1)],
            [FakeDocument(3)],
            [7],
            [FakeDocument(7)],
        )
        self.assertEqual(result["stage"], "reranking")

    def test_dense_hit_lost_during_hybrid_fusion_is_fusion_failure(self) -> None:
        result = _diagnose(
            [FakeDocument(7)],
            [FakeDocument(3)],
            [7],
            [FakeDocument(2)],
        )
        self.assertEqual(result["stage"], "fusion")

    def test_missing_from_both_pipelines_is_recall_failure(self) -> None:
        result = _diagnose([FakeDocument(1)], [FakeDocument(3)], [7])
        self.assertEqual(result["stage"], "recall")


class RetrievalMetricTests(unittest.TestCase):
    def test_first_relevant_rank_uses_the_earliest_matching_page(self) -> None:
        self.assertEqual(first_relevant_rank([3, 7, 9, 7], [7, 8]), 2)
        self.assertIsNone(first_relevant_rank([1, 2, 3], [7]))

    def test_summary_computes_hit_rate_and_mrr_from_ranked_hits(self) -> None:
        cases = [
            {"first_relevant_rank": 1, "latency_ms": 10},
            {"first_relevant_rank": 2, "latency_ms": 20},
            {"first_relevant_rank": None, "latency_ms": 30},
        ]
        result = summarize_pipeline(cases, k=4)
        self.assertEqual(result["evidence_page_hit_rate_at_k"], 0.6667)
        self.assertEqual(result["mrr_at_k"], 0.5)
        self.assertEqual(result["median_latency_ms"], 20)

    def test_development_and_frozen_test_splits_are_both_loaded(self) -> None:
        cases = load_all_retrieval_cases()
        self.assertEqual(len(cases), 28)
        self.assertEqual(sum(case["split"] == "development" for case in cases), 20)
        self.assertEqual(sum(case["split"] == "test" for case in cases), 8)

    def test_manifest_matches_the_local_source_and_label_ranges(self) -> None:
        manifest = json.loads(
            (BASE_DIR / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        source = BASE_DIR / manifest["source_document"]["file"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["source_document"]["sha256"])
        page_count = manifest["source_document"]["page_count"]
        self.assertTrue(
            all(
                1 <= page <= page_count
                for case in load_all_retrieval_cases()
                for page in case["evidence_pages"]
            )
        )

    def test_saved_report_uses_evidence_page_hit_rate_name(self) -> None:
        report = json.loads(
            (BASE_DIR / "evaluation_runs" / "latest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["metric_name"], "Evidence-page Hit Rate@4")
        for pipeline in report["pipelines"]:
            metric_groups = [pipeline["metrics"], *pipeline["metrics_by_split"].values()]
            for metrics in metric_groups:
                self.assertIn("evidence_page_hit_rate_at_k", metrics)
                self.assertNotIn("recall_at_k", metrics)


class GroundedAnswerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            Document(page_content="Gift cards are never allowed.", metadata={"source": "policy.pdf", "page": 14}),
            Document(page_content="Consult compliance before giving value.", metadata={"source": "policy.pdf", "page": 16}),
        ]

    def test_valid_citation_ids_are_mapped_to_server_owned_metadata(self) -> None:
        result = validate_answer_payload(
            {"answerable": True, "answer": "Both rules apply.", "citation_ids": ["S1", "S2"]},
            self.docs,
        )
        self.assertTrue(result["answerable"])
        self.assertEqual([item["page"] for item in result["citations"]], [14, 16])

    def test_unanswerable_response_uses_fixed_abstention(self) -> None:
        result = validate_answer_payload(
            {"answerable": False, "answer": "guess", "citation_ids": []},
            self.docs,
        )
        self.assertFalse(result["answerable"])
        self.assertEqual(result["answer"], REFUSAL_MESSAGE)
        self.assertEqual(result["validation"], "model_abstained")

    def test_answer_with_only_invalid_citations_is_rejected(self) -> None:
        result = validate_answer_payload(
            {"answerable": True, "answer": "unsupported", "citation_ids": ["S9"]},
            self.docs,
        )
        self.assertFalse(result["answerable"])
        self.assertEqual(result["validation"], "missing_valid_citation")

    def test_null_or_blank_answer_is_rejected_even_with_valid_citation(self) -> None:
        for answer in (None, "", "   "):
            with self.subTest(answer=answer):
                result = validate_answer_payload(
                    {"answerable": True, "answer": answer, "citation_ids": ["S1"]},
                    self.docs,
                )
                self.assertFalse(result["answerable"])
                self.assertEqual(result["answer"], REFUSAL_MESSAGE)
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["validation"], "empty_answer")

    def test_string_true_from_compatible_api_is_accepted(self) -> None:
        result = validate_answer_payload(
            {"answerable": "true", "answer": "Supported.", "citation_ids": ["S1"]},
            self.docs,
        )
        self.assertTrue(result["answerable"])


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_overview_reports_real_local_assets(self) -> None:
        response = self.client.get("/api/overview")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["corpus"]["available"])
        self.assertGreater(payload["evaluation"]["case_count"], 0)
        self.assertEqual(payload["evaluation"]["labelled_evidence_count"], 28)
        self.assertEqual(len(payload["pipelines"]), 2)

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_openapi_exposes_grounded_answer_endpoint(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/answer", response.json()["paths"])


if __name__ == "__main__":
    unittest.main()
