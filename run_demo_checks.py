"""Run the three documented answer/citation demo contracts and save evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lab_api import AnswerRequest, answer_question


BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "demo_cases.json"
OUTPUT_PATH = BASE_DIR / "demo_runs" / "latest.json"


def run_demo_checks() -> dict:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        response = answer_question(AnswerRequest(question=case["question"]))
        cited_pages = sorted(
            {
                int(item["page"])
                for item in response["citations"]
                if item.get("page") is not None
            }
        )
        expected_pages = sorted(case["expected_pages"])
        if expected_pages:
            passed = response["answerable"] and set(expected_pages).issubset(cited_pages)
        else:
            passed = not response["answerable"] and not cited_pages
        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_pages": expected_pages,
                "answerable": response["answerable"],
                "answer": response["answer"],
                "cited_pages": cited_pages,
                "validation": response["validation"],
                "passed": bool(passed),
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Three hand-verified demonstration contracts; not an accuracy metric.",
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    report = run_demo_checks()
    print(f"Demo contracts: {report['passed']}/{report['total']} passed")
    for result in report["results"]:
        state = "PASS" if result["passed"] else "FAIL"
        print(f"  {state} {result['id']}: cited pages={result['cited_pages']}")
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
