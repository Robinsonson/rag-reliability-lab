"""Evaluate an already-ingested synthetic corpus; never import or edit policies.

Default: live retrieval only. --answers explicitly enables billable model-backed
answers. --offline REPORT recomputes deterministic metrics with no network.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.cookiejar
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LIMITATION = (
    "Title and literal excerpt coverage, source scope/version/excerpt checks and citation-label checks "
    "are deterministic diagnostics, NOT semantic answer accuracy, factual entailment, policy-change "
    "safety certification, or a 10,000-document / 100 MiB capacity benchmark. Latencies exclude login "
    "and metadata-validation requests. Small-sample p95 values are descriptive only."
)


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def percentile(values, percentile_value):
    """Linear interpolation on sorted observations; empty input is unavailable."""
    if not 0 <= percentile_value <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    clean = sorted(float(value) for value in values)
    if not clean:
        return None
    if any(not math.isfinite(value) or value < 0 for value in clean):
        raise ValueError("Latencies must be finite and nonnegative")
    position = (len(clean) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)


def literal_coverage(expected, evidence):
    matches = []
    for target in expected:
        same_title = [e for e in evidence if normalize(e.get("title", "")) == normalize(target["title"])]
        found = [e for e in same_title if all(normalize(fragment) in normalize(e.get("text", ""))
                                             for fragment in target.get("text_contains", []))]
        matches.append({"title": target["title"], "required_fragments": target.get("text_contains", []),
                        "title_found": bool(same_title), "literal_excerpt_found": bool(found),
                        "matched_source_ids": [e.get("id") for e in found]})
    return {"expected_items": len(expected), "title_matches": sum(m["title_found"] for m in matches),
            "literal_excerpt_matches": sum(m["literal_excerpt_found"] for m in matches), "matches": matches}


def source_invariants(sources, observation):
    """Check server-owned source metadata; do not infer semantic correctness."""
    actor = observation.get("actor", {})
    before = observation.get("policies_before", {})
    after = observation.get("policies_after", {})
    revisions = observation.get("revisions", {})
    violations, concurrent, checked = [], set(), 0
    for source in sources:
        checked += 1
        identifier = source.get("id", "<missing>")

        def fail(code):
            violations.append({"source_id": identifier, "code": code})

        policy_id, revision_id = source.get("policy_id"), source.get("revision_id")
        if not all(source.get(key) for key in ["id", "policy_id", "revision_id", "block_id", "scope", "title", "text"]):
            fail("required_source_field_missing")
        if actor.get("role") != "admin" and source.get("scope") not in actor.get("scopes", []):
            fail("outside_account_scope")
        old, current = before.get(policy_id), after.get(policy_id)
        if old is None or current is None:
            fail("policy_missing_from_accessible_catalog")
        else:
            if source.get("scope") != current.get("scope") or source.get("title") != current.get("title"):
                fail("policy_metadata_mismatch")
            old_revision, current_revision = old.get("published_revision_id"), current.get("published_revision_id")
            if old_revision != current_revision:
                concurrent.add(policy_id)
                if revision_id not in {old_revision, current_revision}:
                    fail("revision_matches_neither_observed_published_version")
            elif revision_id != current_revision:
                fail("source_revision_is_not_published_version")
        revision = revisions.get(revision_id)
        if not revision or revision.get("unavailable"):
            fail("revision_metadata_unavailable")
        else:
            if revision.get("id") != revision_id or revision.get("policy_id") != policy_id:
                fail("revision_ownership_mismatch")
            if type(source.get("revision_number")) is not int or source.get("revision_number") != revision.get("number"):
                fail("revision_number_mismatch")
            block = next((block for block in revision.get("blocks", []) if block.get("id") == source.get("block_id")), None)
            if block is None:
                fail("canonical_block_missing")
            elif not source.get("text") or source["text"] not in block.get("text", ""):
                fail("excerpt_is_not_canonical_substring")
    return {"sources_checked": checked, "violations": violations,
            "concurrent_publication_policy_ids": sorted(concurrent),
            "all_applicable_checks_passed": not violations,
            "published_snapshot_check": "inconclusive_concurrent_publication" if concurrent else "stable_between_observations"}


def answer_shape_checks(result):
    violations = []
    if type(result.get("answerable")) is not bool or not isinstance(result.get("answer"), str):
        violations.append("invalid_answer_shape")
    citations = result.get("citations", [])
    if not isinstance(citations, list) or any(not isinstance(item, dict) for item in citations):
        return {"passed": False, "violations": violations + ["invalid_citation_shape"]}
    if result.get("answerable") is True:
        labels = [c.get("citation_label") for c in citations]
        inline = set(re.findall(r"\[(S\d+)\]", result.get("answer", "")))
        if not citations or not inline or set(labels) != inline or len(set(labels)) != len(labels):
            violations.append("inline_citation_labels_do_not_match_sources")
    elif result.get("answerable") is False and citations:
        violations.append("unanswerable_result_has_citations")
    return {"passed": not violations, "violations": violations}


def score_report(dataset, captured):
    if captured.get("dataset") != dataset["dataset"]:
        raise ValueError("Captured report and case dataset identifiers differ")
    rows = captured.get("raw_cases", [])
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Captured case IDs must be unique")
    expected_total = titles = excerpts = 0
    retrieval_ms, answer_ms, scored_cases = [], [], []
    errors = invariant_violations = answer_shape_failures = 0
    concurrent_cases = 0
    for case in dataset["cases"]:
        row = by_id.get(case["id"], {})
        operation = row.get("retrieval", {"error": "not_captured"})
        coverage = literal_coverage(case["expected"], operation.get("evidence", []))
        expected_total += coverage["expected_items"]
        titles += coverage["title_matches"]
        excerpts += coverage["literal_excerpt_matches"]
        scored = {"id": case["id"], "retrieval_coverage": coverage}
        for key, latencies in [("retrieval", retrieval_ms), ("answer", answer_ms)]:
            observed = row.get(key)
            if observed is None:
                if key == "retrieval":
                    errors += 1
                    scored[key + "_error"] = "not_captured"
                continue
            if observed.get("latency_ms") is not None:
                latencies.append(observed["latency_ms"])
            if observed.get("error"):
                errors += 1
                scored[key + "_error"] = observed["error"]
                continue
            result = observed.get("result", {})
            sources = observed.get("evidence", []) if key == "retrieval" else result.get("citations", [])
            if not isinstance(sources, list) or any(not isinstance(source, dict) for source in sources):
                errors += 1
                scored[key + "_error"] = "invalid_sources_shape"
                continue
            invariant = source_invariants(sources, observed)
            invariant_violations += len(invariant["violations"])
            concurrent_cases += bool(invariant["concurrent_publication_policy_ids"])
            scored[key + "_source_invariants"] = invariant
            if key == "answer":
                scored["answer_shape_checks"] = answer_shape_checks(result)
                answer_shape_failures += not scored["answer_shape_checks"]["passed"]
        scored_cases.append(scored)
    return {
        "schema_version": 1, "dataset": dataset["dataset"], "synthetic": True, "limitations": LIMITATION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture": captured.get("capture", {}),
        "summary": {
            "case_count": len(dataset["cases"]), "expected_retrieval_items": expected_total,
            "retrieval_title_coverage": titles / expected_total if expected_total else None,
            "retrieval_literal_excerpt_coverage": excerpts / expected_total if expected_total else None,
            "operation_errors": errors, "source_invariant_violation_count": invariant_violations,
            "answer_shape_failure_count": answer_shape_failures,
            "operations_with_concurrent_publication": concurrent_cases,
            "retrieval_latency_ms": {"count": len(retrieval_ms), "p50": percentile(retrieval_ms, 50), "p95": percentile(retrieval_ms, 95)},
            "answer_latency_ms": {"count": len(answer_ms), "p50": percentile(answer_ms, 50), "p95": percentile(answer_ms, 95)},
        },
        "cases": scored_cases, "raw_cases": rows,
    }


class RequestFailure(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RequestFailure("Unexpected HTTP redirect; use the exact API base URL")


class APIClient:
    def __init__(self, base_url, timeout=180):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an http(s) base URL without credentials, query or fragment")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.csrf = None
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()), NoRedirect())

    def request(self, path, payload=None):
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            if self.csrf:
                headers["X-CSRF-Token"] = self.csrf
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
                if len(raw) > 32 * 1024 * 1024:
                    raise RequestFailure("Response exceeds the evaluator's 32 MiB response limit")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            # Never put provider response bodies, cookies or passwords in a report.
            raise RequestFailure(f"HTTP {exc.code} from {path.split('?')[0]}") from None
        except (urllib.error.URLError, TimeoutError):
            raise RequestFailure("API network request failed or timed out") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RequestFailure("API returned invalid JSON") from None

    def login(self, username, password):
        result = self.request("/api/v2/login", {"username": username, "password": password})
        if not result.get("csrf_token"):
            raise RequestFailure("Login returned no CSRF token")
        self.csrf = result["csrf_token"]

    def policies(self):
        result, offset = {}, 0
        for _ in range(1000):
            batch = self.request(f"/api/v2/policies?limit=100&offset={offset}").get("policies", [])
            for policy in batch:
                result[policy["id"]] = {key: policy.get(key) for key in ("id", "title", "scope", "published_revision_id")}
            if len(batch) < 100:
                return result
            offset += len(batch)
        raise RequestFailure("Policy catalog pagination exceeded the evaluation bound")


def capture_operation(client, question, kind):
    observation = {}
    try:
        session = client.request("/api/v2/session")
        client.csrf = session["csrf_token"]
        user = session["user"]
        observation["actor"] = {"role": user["role"], "scopes": user["scopes"]}
        observation["policies_before"] = client.policies()
        start = time.perf_counter()
        try:
            response = client.request("/api/v2/retrieval" if kind == "retrieval" else "/api/v2/answers", {"question": question})
        finally:
            observation["latency_ms"] = (time.perf_counter() - start) * 1000
        if kind == "retrieval":
            sources = observation["evidence"] = response["evidence"]
        else:
            observation["result"] = response["result"]
            observation["thread_id"] = response.get("thread_id")
            sources = response["result"].get("citations", [])
        observation["policies_after"] = client.policies()
        observation["revisions"] = {}
        revision_ids = {source["revision_id"] for source in sources}
        for revision_id in revision_ids:
            revision_path = "/api/v2/revisions/" + urllib.parse.quote(str(revision_id), safe="")
            try:
                revision = client.request(revision_path)
                selected_blocks = {source["block_id"] for source in sources if source["revision_id"] == revision_id}
                observation["revisions"][revision_id] = {
                    "id": revision["id"], "policy_id": revision["policy_id"], "number": revision["number"],
                    "blocks": [{"id": block["id"], "text": block["text"]} for block in revision["blocks"] if block["id"] in selected_blocks],
                }
            except RequestFailure:
                observation["revisions"][revision_id] = {"unavailable": True}
    except RequestFailure as exc:
        observation["error"] = str(exc)
    except (KeyError, TypeError):
        observation["error"] = "invalid_api_response_shape"
    return observation


def load_dataset(path):
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 1 or dataset.get("synthetic") is not True:
        raise ValueError("Expected a version 1 synthetic evaluation dataset")
    cases = dataset.get("cases", [])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation cases must be nonempty and have unique IDs")
    for case in cases:
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("Each case must have a question")
        if not isinstance(case.get("expected"), list):
            raise ValueError("Each case must have an expected-items list")
        for expected in case["expected"]:
            if not expected.get("title") or any(not isinstance(s, str) or not s for s in expected.get("text_contains", [])):
                raise ValueError("Expected items require titles and nonempty literal fragments")
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", type=Path, default=REPOSITORY_ROOT / "evaluation_v2_cases.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "benchmark-results" / "v2.json")
    parser.add_argument("--answers", action="store_true", help="Also call real model-backed answers; may incur provider charges")
    parser.add_argument("--offline", type=Path, help="Re-score a saved report without login, network or model calls")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if args.offline and args.answers:
            raise ValueError("--offline and --answers cannot be combined")
        dataset = load_dataset(args.cases)
        if args.offline:
            captured = json.loads(args.offline.read_text(encoding="utf-8"))
        else:
            username, password = os.getenv("POLICY_EVAL_USERNAME"), os.getenv("POLICY_EVAL_PASSWORD")
            if not username or not password:
                raise ValueError("Set POLICY_EVAL_USERNAME and POLICY_EVAL_PASSWORD in the environment")
            client = APIClient(args.base_url, args.timeout)
            client.login(username, password)
            available = {(p["title"], p["scope"]) for p in client.policies().values() if p["published_revision_id"]}
            missing = [fixture["title"] for fixture in dataset.get("fixtures", []) if (fixture["title"], fixture["scope"]) not in available]
            if missing:
                raise ValueError("Required published fixtures are absent or inaccessible; manually upload the case-file fixtures and verify account scopes: " + "; ".join(missing))
            captured = {"dataset": dataset["dataset"], "capture": {"base_url": args.base_url, "answers_enabled": args.answers,
                        "mode": "read_only_policy_queries", "started_at_utc": datetime.now(timezone.utc).isoformat()}, "raw_cases": []}
            try:
                for case in dataset["cases"]:
                    print(f"Evaluating {case['id']}", file=sys.stderr)
                    row = {"id": case["id"], "question": case["question"], "retrieval": capture_operation(client, case["question"], "retrieval")}
                    if args.answers:
                        row["answer"] = capture_operation(client, case["question"], "answer")
                    captured["raw_cases"].append(row)
            finally:
                try:
                    client.request("/api/v2/logout", {})
                except RequestFailure:
                    pass
        report = score_report(dataset, captured)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved deterministic diagnostic report: {args.output.resolve()}")
        summary = report["summary"]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["operation_errors"] or summary["source_invariant_violation_count"] or summary["answer_shape_failure_count"] else 0
    except (ValueError, RequestFailure, OSError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
