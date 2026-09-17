"""End-to-end API contracts using persistent SQLite and real LangGraph execution.

Embedding and model responses are injected to avoid external calls. These tests
verify workflow correctness, not model accuracy or PostgreSQL performance.
"""
import hashlib
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from policy_lab.api import create_app
from policy_lab.auth import hash_password
from policy_lab.config import Settings
from policy_lab.parsing import DocumentParser
from policy_lab.repository import Repository
from policy_lab.storage import LocalStorage


class FixtureEmbedder:
    def encode(self, texts):
        return [[float(v) / 255 for v in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


class FixtureModel:
    def complete(self, task, payload):
        if task == "assess_evidence":
            return {"sufficient": bool(payload["evidence"]), "relevant_ids": [e["id"] for e in payload["evidence"]], "rewritten_query": ""}
        if task == "answer":
            return {"answer": payload["sources"][0]["text"] + " [S1]", "citation_ids": ["S1"]}
        if task == "select_clause":
            return {"selected_id": payload["candidates"][0]["id"], "question": ""}
        if task == "draft_clause":
            return {"before_span": "500", "after_span": "800", "reason": "Adjust requested lodging limit", "question": ""}
        raise AssertionError(task)


class ApiV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password = "fixture-password-123"
        cls.password_hash = hash_password(cls.password)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = Settings(database_url="sqlite:///" + (root / "test.db").as_posix(), data_dir=root,
                                 embedding_dimension=8, enable_reranker=False, max_upload_bytes=4096)
        self.repo = Repository(self.settings, FixtureEmbedder())
        self.repo.initialize()
        self.admin = self.repo.create_user("admin", self.password_hash, "admin", ["general"])
        self.manager = self.repo.create_user("boss", self.password_hash, "manager", ["general"])
        self.reader = self.repo.create_user("reader", self.password_hash, "reader", ["general"])
        self.store = LocalStorage(root / "objects")
        self.app = create_app(self.settings, self.repo, self.store, llm=FixtureModel())
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.repo.engine.dispose()
        self.directory.cleanup()

    def login(self, name="boss"):
        response = self.client.post("/api/v2/login", json={"username": name, "password": self.password})
        self.assertEqual(response.status_code, 200, response.text)
        return {"X-CSRF-Token": response.json()["csrf_token"]}

    def upload(self, scope="general", content="国内出差住宿标准为每晚 500 元。", headers=None):
        return self.client.post("/api/v2/uploads", data={"title": "差旅管理制度", "scope": scope},
                                files={"file": ("policy.txt", content.encode(), "text/plain")},
                                headers=headers or self.login())

    def process(self):
        self.assertTrue(self.repo.process_one_job(storage=self.store, parser=DocumentParser(self.settings)))

    def test_session_cookie_csrf_and_old_routes_closed(self):
        self.assertEqual(self.client.get("/api/v2/policies").status_code, 401)
        self.assertEqual(self.client.get("/api/documents").status_code, 404)
        headers = self.login()
        self.assertIn("httponly", self.client.cookies.get("policy_session") and
                      self.client.post("/api/v2/login", json={"username":"boss","password":self.password}).headers.get("set-cookie", "").lower())
        self.assertEqual(self.client.post("/api/v2/changes", json={"instruction": "改成800元"}).status_code, 403)
        self.assertEqual(self.client.post("/api/v2/changes", json={"instruction": "改成800元"},
                                         headers={**headers, "Origin": "https://attacker.invalid"}).status_code, 403)
        self.assertNotIn("password_hash", self.client.get("/api/v2/session").text)

    def test_upload_is_async_and_scope_filters_every_read(self):
        headers = self.login()
        uploaded = self.upload(headers=headers)
        self.assertEqual(uploaded.status_code, 202, uploaded.text)
        job = uploaded.json()
        policy = self.client.get("/api/v2/policies").json()["policies"][0]
        self.assertIsNone(policy["published_revision_id"])
        self.process()
        self.assertEqual(self.client.get("/api/v2/jobs/" + job["id"]).json()["status"], "succeeded")
        self.assertEqual(self.upload(scope="finance", headers=headers).status_code, 403)
        self.repo.update_user(self.manager["id"], "manager", ["other"], True)
        self.assertEqual(self.client.get("/api/v2/policies").json()["policies"], [])
        self.assertEqual(self.client.get("/api/v2/revisions/" + job["revision_id"]).status_code, 403)
        self.assertEqual(self.client.get("/api/v2/revisions/" + job["revision_id"] + "/original").status_code, 403)

    def test_review_publish_export_and_immutable_original(self):
        headers = self.login()
        original = "国内出差住宿标准为每晚 500 元。"
        first = self.upload(content=original, headers=headers).json()
        self.process()
        proposed = self.client.post("/api/v2/changes", json={"instruction": "把国内出差住宿标准改成800元"}, headers=headers)
        self.assertEqual(proposed.status_code, 200, proposed.text)
        payload = proposed.json()
        self.assertEqual(payload["status"], "interrupted")
        self.assertIn("800", payload["interrupts"][0]["proposal"]["after"])
        thread = payload["thread_id"]
        self.assertEqual(self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":"true"}, headers=headers).status_code, 422)
        self.assertEqual(self.client.post(f"/api/v2/changes/{thread}/resume", json={"choice_id":"wrong"}, headers=headers).status_code, 400)
        approved = self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":True}, headers=headers)
        self.assertEqual(approved.status_code, 200, approved.text)
        second = approved.json()["result"]["job"]
        self.assertEqual(self.repo.get_policy(self.manager, first["policy_id"])["published_revision_id"], first["revision_id"])
        self.process()
        self.assertEqual(self.repo.get_job(self.manager, second["id"])["status"], "succeeded")
        export = self.client.get(f"/api/v2/revisions/{second['revision_id']}/export")
        self.assertIn("800", export.text)
        attachment = self.client.get(f"/api/v2/revisions/{second['revision_id']}/original")
        self.assertEqual(attachment.content, original.encode())
        answer = self.client.post("/api/v2/answers", json={"question":"住宿标准是多少？"}, headers=headers)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertIn("800", answer.json()["result"]["answer"])
        self.assertEqual(answer.json()["result"]["citations"][0]["revision_id"], second["revision_id"])
        again = self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":True}, headers=headers)
        self.assertEqual(again.json()["result"]["job"]["id"], second["id"])

    def test_reader_cannot_edit_and_foreign_thread_denied(self):
        headers = self.login()
        self.upload(headers=headers)
        self.process()
        response = self.client.post("/api/v2/changes", json={"instruction":"标准从500调整成800元"}, headers=headers)
        thread = response.json()["thread_id"]
        reader_headers = self.login("reader")
        self.assertEqual(self.client.post("/api/v2/changes", json={"instruction":"改成800元"}, headers=reader_headers).status_code, 403)
        admin_headers = self.login("admin")
        self.assertEqual(self.client.get(f"/api/v2/changes/{thread}").status_code, 403)
        self.assertEqual(self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":True}, headers=admin_headers).status_code, 403)

    def test_revocation_hides_saved_review_and_blocks_publish(self):
        headers = self.login()
        self.upload(headers=headers)
        self.process()
        response = self.client.post("/api/v2/changes", json={"instruction":"标准从500调整成800元"}, headers=headers)
        thread = response.json()["thread_id"]
        self.repo.update_user(self.manager["id"], "manager", ["other"], True)
        self.assertEqual(self.client.get(f"/api/v2/changes/{thread}").status_code, 403)
        self.assertEqual(self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":True}, headers=headers).status_code, 403)

    def test_size_limit_and_unauthorized_account_creation(self):
        headers = self.login()
        self.assertEqual(self.upload(content="x" * 4097, headers=headers).status_code, 413)
        self.assertEqual(self.client.post("/api/v2/users", json={"username":"outsider", "password":self.password,
                "role":"admin", "scopes":["general"]}, headers=headers).status_code, 403)

    def test_answer_checkpoint_cannot_be_replayed_as_change_after_scope_revocation(self):
        headers = self.login()
        self.upload(headers=headers)
        self.process()
        answer = self.client.post("/api/v2/answers", json={"question":"住宿标准是多少"}, headers=headers)
        self.assertEqual(answer.status_code, 200, answer.text)
        thread = answer.json()["thread_id"]
        self.repo.update_user(self.manager["id"], "manager", ["other"], True)
        response = self.client.get(f"/api/v2/changes/{thread}")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("500", response.text)
        response = self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":True}, headers=headers)
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("500", response.text)

    def test_publish_commit_then_crash_is_recoverable_through_public_api(self):
        headers = self.login()
        self.upload(headers=headers)
        self.process()
        response = self.client.post("/api/v2/changes", json={"instruction":"住宿500改成800"}, headers=headers)
        thread = response.json()["thread_id"]
        original_publish = self.repo.publish_proposal
        created_jobs = []
        def crash_after_commit(*args):
            created_jobs.append(original_publish(*args))
            raise RuntimeError("Injected process crash after committing publication job")
        with patch.object(self.repo, "publish_proposal", side_effect=crash_after_commit):
            with self.assertRaises(RuntimeError):
                self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":True}, headers=headers)
        status = self.client.get(f"/api/v2/changes/{thread}")
        self.assertEqual(status.json()["status"], "recoverable")
        resumed = self.client.post(f"/api/v2/changes/{thread}/resume", json={"approve":True}, headers=headers)
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.assertEqual(resumed.json()["result"]["job"]["id"], created_jobs[0]["id"])
        self.process()
        self.assertEqual(self.repo.get_job(self.manager, created_jobs[0]["id"])["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
