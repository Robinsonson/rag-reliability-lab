"""Offline metric checks; these tests never connect to an API or an LLM."""
from copy import deepcopy
import json

import pytest

from scripts import evaluate_policy_v2 as evaluation


def example():
    source = {"id": "r1:b1", "policy_id": "p1", "revision_id": "r1", "revision_number": 1,
              "block_id": "b1", "scope": "hr", "title": "Synthetic Travel", "text": "General employees: CNY 500.", "citation_label": "S1"}
    policy = {"id": "p1", "scope": "hr", "title": "Synthetic Travel", "published_revision_id": "r1"}
    observation = {"actor": {"role": "reader", "scopes": ["hr"]}, "policies_before": {"p1": policy},
                   "policies_after": {"p1": deepcopy(policy)}, "evidence": [source], "latency_ms": 25,
                   "revisions": {"r1": {"id": "r1", "policy_id": "p1", "number": 1,
                                          "blocks": [{"id": "b1", "text": "Introduction. General employees: CNY 500. Exceptions require approval."}]}}}
    dataset = {"schema_version": 1, "dataset": "offline-fixture", "synthetic": True,
               "cases": [{"id": "a", "question": "Domestic lodging limit?", "expected": [{"title": "Synthetic Travel", "text_contains": ["general employees", "500"]}]}]}
    return source, observation, dataset


def test_literal_coverage_requires_title_and_fragments_in_same_excerpt():
    targets = [{"title": "政策甲", "text_contains": ["500", "普通员工"]}]
    sources = [{"id": "a", "title": "政策甲", "text": "普通员工未提供金额"},
               {"id": "b", "title": "政策甲", "text": "承包商500元"},
               {"id": "c", "title": "政策乙", "text": "普通员工500元"}]
    score = evaluation.literal_coverage(targets, sources)
    assert score["title_matches"] == 1
    assert score["literal_excerpt_matches"] == 0
    sources.append({"id": "d", "title": "政策甲", "text": "普通员工５００元"})
    assert evaluation.literal_coverage(targets, sources)["matches"][0]["matched_source_ids"] == ["d"]


def test_percentiles_are_interpolated_and_empty_is_unavailable():
    assert evaluation.percentile([30, 10, 20], 50) == 20
    assert evaluation.percentile([30, 10, 20], 95) == 29
    assert evaluation.percentile([17], 95) == 17
    assert evaluation.percentile([], 95) is None
    with pytest.raises(ValueError):
        evaluation.percentile([float("nan")], 95)


def test_valid_original_excerpt_and_metadata_pass_mechanical_checks():
    source, observation, _ = example()
    result = evaluation.source_invariants([source], observation)
    assert result["sources_checked"] == 1
    assert result["violations"] == []
    assert result["published_snapshot_check"] == "stable_between_observations"


@pytest.mark.parametrize("field,value,expected", [
    ("scope", "finance", "outside_account_scope"),
    ("revision_number", 9, "revision_number_mismatch"),
    ("text", "Unsupported 800 CNY", "excerpt_is_not_canonical_substring"),
    ("block_id", "not-found", "canonical_block_missing"),
])
def test_scope_version_and_excerpt_failures_are_reported(field, value, expected):
    source, observation, _ = example()
    source[field] = value
    codes = {v["code"] for v in evaluation.source_invariants([source], observation)["violations"]}
    assert expected in codes


def test_concurrent_publication_is_inconclusive_instead_of_false_stale_failure():
    source, observation, _ = example()
    observation["policies_after"]["p1"]["published_revision_id"] = "r2"
    result = evaluation.source_invariants([source], observation)
    assert result["published_snapshot_check"] == "inconclusive_concurrent_publication"
    assert result["concurrent_publication_policy_ids"] == ["p1"]
    assert not result["violations"]


def test_stable_current_revision_rejects_stale_source():
    source, observation, _ = example()
    observation["policies_before"]["p1"]["published_revision_id"] = "r2"
    observation["policies_after"]["p1"]["published_revision_id"] = "r2"
    codes = {v["code"] for v in evaluation.source_invariants([source], observation)["violations"]}
    assert "source_revision_is_not_published_version" in codes


def test_citation_label_check_does_not_claim_semantic_correctness():
    source, _, _ = example()
    # A wrong amount can pass a label check: this deliberately documents the limit.
    result = {"answerable": True, "answer": "The limit is CNY 999 [S1].", "citations": [source]}
    assert evaluation.answer_shape_checks(result)["passed"] is True
    result["answer"] = "The limit is CNY 999 [S99]."
    assert evaluation.answer_shape_checks(result)["passed"] is False
    assert "NOT semantic answer accuracy" in evaluation.LIMITATION


def test_report_has_raw_observations_counts_and_no_accuracy_metric():
    _, observation, dataset = example()
    raw = {"dataset": dataset["dataset"], "raw_cases": [{"id": "a", "retrieval": observation}]}
    report = evaluation.score_report(dataset, raw)
    assert report["summary"]["retrieval_literal_excerpt_coverage"] == 1
    assert report["summary"]["retrieval_latency_ms"]["p95"] == 25
    assert report["summary"]["answer_latency_ms"]["p95"] is None
    assert report["raw_cases"] == raw["raw_cases"]
    assert not any("accuracy" in key for key in report["summary"])


def test_missing_case_counts_as_error_and_zero_coverage():
    _, _, dataset = example()
    report = evaluation.score_report(dataset, {"dataset": dataset["dataset"], "raw_cases": []})
    assert report["summary"]["retrieval_literal_excerpt_coverage"] == 0
    assert report["summary"]["operation_errors"] == 1


def test_offline_cli_does_not_login_or_connect(tmp_path, monkeypatch):
    _, observation, dataset = example()
    cases = tmp_path / "cases.json"
    raw = tmp_path / "saved.json"
    output = tmp_path / "metrics.json"
    cases.write_text(json.dumps(dataset), encoding="utf-8")
    raw.write_text(json.dumps({"dataset": dataset["dataset"], "raw_cases": [{"id": "a", "retrieval": observation}]}), encoding="utf-8")
    monkeypatch.delenv("POLICY_EVAL_USERNAME", raising=False)
    monkeypatch.delenv("POLICY_EVAL_PASSWORD", raising=False)

    def fail_if_network(*args, **kwargs):
        raise AssertionError("Offline evaluation must not construct a network client")

    monkeypatch.setattr(evaluation, "APIClient", fail_if_network)
    assert evaluation.main(["--offline", str(raw), "--cases", str(cases), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["source_invariant_violation_count"] == 0


def test_live_client_never_accepts_credentials_in_url():
    with pytest.raises(ValueError, match="without credentials"):
        evaluation.APIClient("https://user:password@example.com")


def test_default_bilingual_fixture_cases_are_valid_and_do_not_touch_frozen_cases():
    dataset = evaluation.load_dataset(evaluation.REPOSITORY_ROOT / "evaluation_v2_cases.json")
    assert len(dataset["cases"]) == 9
    assert len(dataset["fixtures"]) == 2
    assert dataset["synthetic"] is True
    assert {fixture["scope"] for fixture in dataset["fixtures"]} == {"hr", "finance"}
