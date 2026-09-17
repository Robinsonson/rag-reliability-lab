"""Security/publication invariants using SQLite and explicit test-only vectors."""
from dataclasses import replace
import hashlib
import io
import time

import pytest
import sqlalchemy as sa

from policy_lab.config import Settings
from policy_lab.indexing import lexical_tokens
from policy_lab.repository import Repository, ConflictError
from policy_lab.storage import LocalStorage


class TestEmbedder:
    __test__ = False
    """Only a deterministic test fixture: not a production semantic model."""
    def __init__(self):
        self.calls = 0
        self.fail = False

    def encode(self, texts):
        self.calls += len(texts)
        if self.fail:
            raise RuntimeError("Injected embedding outage")
        result = []
        for text in texts:
            vector = [0.] * 16
            for token in lexical_tokens(text):
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % 16] += 1
            result.append(vector)
        return result


@pytest.fixture
def setup(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'db.sqlite'}", data_dir=tmp_path,
        embedding_dimension=16, enable_reranker=False, model_cache_dir=tmp_path / "models")
    embedder = TestEmbedder()
    repo = Repository(settings, embedder=embedder)
    repo.initialize()
    admin = repo.create_user("admin", "unused", "admin", [])
    manager = repo.create_user("hr_manager", "unused", "manager", ["hr"])
    reader = repo.create_user("reader", "unused", "reader", ["hr"])
    outsider = repo.create_user("finance", "unused", "manager", ["finance"])
    storage = LocalStorage(tmp_path / "objects")
    storage.ensure_bucket()
    yield repo, storage, embedder, admin, manager, reader, outsider
    repo.engine.dispose()


def upload(repo, storage, actor, text="# Travel policy\n\n国内出差住宿上限为500元。\n\nMeal allowance is 100 yuan.", scope="hr", replaces=None):
    raw = text.encode()
    sha = hashlib.sha256(raw).hexdigest()
    key = "uploads/" + sha[:16] + (replaces[:8] if replaces else "")
    storage.put_file(key, io.BytesIO(raw))
    return repo.enqueue_upload(actor, "Travel policy", scope, key, "policy.md", sha, len(raw), replaces=replaces)


def publish_upload(setup):
    repo, storage, _, _, manager, *_ = setup
    job = upload(repo, storage, manager)
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, job["id"])["status"] == "succeeded"
    return job


def evidence(repo, actor, policy_id):
    current = repo.get_policy(actor, policy_id)["published_revision_id"]
    revision = repo.get_revision(actor, current)
    block = next(b for b in revision["blocks"] if "500" in b["text"])
    return dict(policy_id=policy_id, revision_id=current, block_id=block["id"], text=block["text"])


def test_upload_durable_acl_session_and_duplicate(setup):
    repo, storage, _, admin, manager, reader, outsider = setup
    job = publish_upload(setup)
    assert not repo.process_one_job(storage)
    assert "500" in repo.search(reader, "国内出差住宿")[0]["text"]
    assert repo.search(outsider, "国内出差住宿") == []
    with pytest.raises(PermissionError):
        repo.get_revision(outsider, job["revision_id"])
    with pytest.raises(PermissionError):
        repo.search(reader, "住宿", editable=True)
    with pytest.raises(ConflictError):
        repo.enqueue_upload(manager, "Duplicate", "hr", "duplicate", "copy.md", hashlib.sha256("# Travel policy\n\n国内出差住宿上限为500元。\n\nMeal allowance is 100 yuan.".encode()).hexdigest(), 100)
    repo.create_session("token", reader["id"], time.time() + 30)
    assert repo.get_session("token")["id"] == reader["id"]
    repo.update_user(reader["id"], "reader", ["hr"], False)
    assert repo.get_session("token") is None
    with pytest.raises(PermissionError):
        repo.search(reader, "住宿")


def test_proposal_index_before_publish_snapshot_and_rollback(setup):
    repo, storage, embedder, _, manager, reader, outsider = setup
    job = publish_upload(setup)
    ev = evidence(repo, manager, job["policy_id"])
    original_blocks = repo.get_revision(manager, ev["revision_id"])["blocks"]
    pinned = repo.snapshot(reader)
    proposal = repo.create_proposal(manager, "住宿改为800", ev, "国内出差住宿上限为800元。", "调整上限", request_key="change-1")
    same = repo.create_proposal(manager, "住宿改为800", ev, "国内出差住宿上限为800元。", "调整上限", request_key="change-1")
    assert same["id"] == proposal["id"]
    before_calls = embedder.calls
    publication = repo.publish_proposal(manager, proposal["id"])
    assert repo.publish_proposal(manager, proposal["id"])["id"] == publication["id"]
    assert repo.get_policy(reader, job["policy_id"])["published_revision_id"] == ev["revision_id"]
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, publication["id"])["status"] == "succeeded"
    assert embedder.calls - before_calls == 1  # Only changed paragraph is embedded; heading/other block reused.
    updated = repo.get_revision(reader, publication["revision_id"])
    assert [b["id"] for b in updated["blocks"]] == [b["id"] for b in original_blocks]
    assert "800" in repo.search(reader, "住宿上限")[0]["text"]
    assert "500" in repo.search(reader, "住宿上限", snapshot=pinned)[0]["text"]
    assert repo.search(outsider, "住宿上限", snapshot=pinned) == []
    rollback = repo.rollback(manager, job["policy_id"], ev["revision_id"], updated["id"])
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, rollback["id"])["status"] == "succeeded"
    assert "500" in repo.search(reader, "住宿上限")[0]["text"]


def test_failure_and_conflict_leave_published_revision_unchanged(setup):
    repo, storage, embedder, _, manager, reader, _ = setup
    job = publish_upload(setup)
    ev = evidence(repo, manager, job["policy_id"])
    first = repo.create_proposal(manager, "change", ev, "国内出差住宿上限为800元。", "first")
    second = repo.create_proposal(manager, "change", ev, "国内出差住宿上限为900元。", "second")
    j1 = repo.publish_proposal(manager, first["id"])
    j2 = repo.publish_proposal(manager, second["id"])
    embedder.fail = True
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, j1["id"])["status"] == "failed"
    assert repo.get_policy(reader, job["policy_id"])["published_revision_id"] == ev["revision_id"]
    embedder.fail = False
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, j2["id"])["status"] == "succeeded"
    repo.retry_job(manager, j1["id"])
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, j1["id"])["status"] == "failed"
    assert "900" in repo.search(reader, "住宿上限")[0]["text"]


def test_revoked_manager_cannot_publish_and_thread_ownership(setup):
    repo, storage, _, admin, manager, reader, outsider = setup
    job = publish_upload(setup)
    ev = evidence(repo, manager, job["policy_id"])
    proposal = repo.create_proposal(manager, "change", ev, "800元", "reason")
    publication = repo.publish_proposal(manager, proposal["id"])
    repo.create_thread(manager, "thread-1")
    assert repo.authorize_thread(manager, "thread-1")
    with pytest.raises(PermissionError):
        repo.authorize_thread(outsider, "thread-1")
    repo.update_user(manager["id"], "manager", ["finance"], True)
    assert repo.process_one_job(storage)
    assert repo.get_job(admin, publication["id"])["status"] == "failed"
    assert repo.get_policy(reader, job["policy_id"])["published_revision_id"] == ev["revision_id"]


def test_expired_lease_recovery_and_lost_worker_cannot_write(setup):
    repo, storage, _, _, manager, *_ = setup
    queued = upload(repo, storage, manager)
    old = repo._claim_job()
    with repo.engine.begin() as conn:
        conn.execute(repo.jobs.update().where(repo.jobs.c.id == old["id"]).values(lease_until=time.time() - 1))
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, queued["id"])["attempts"] == 2
    assert repo.get_job(manager, queued["id"])["status"] == "succeeded"
    with pytest.raises(ConflictError):
        repo._progress(old)
    with repo._tx() as conn:
        with pytest.raises(ConflictError):
            repo._check_lease(conn, old)


def test_reader_cannot_forge_snapshot_or_actor_and_staged_revision(setup):
    repo, storage, _, admin, manager, reader, outsider = setup
    job = upload(repo, storage, manager)
    with pytest.raises(PermissionError):
        repo.get_revision(reader, job["revision_id"])
    assert repo.search(reader, "住宿", snapshot={job["policy_id"]: job["revision_id"]}) == []
    forged = dict(reader, role="admin", scopes=["finance"])
    with pytest.raises(PermissionError):
        repo.enqueue_upload(forged, "bad", "finance", "key", "a.txt", "abc", 10)


def test_original_checksum_limits_and_replacement_base(setup):
    repo, storage, _, admin, manager, reader, _ = setup
    first = publish_upload(setup)
    replacement = upload(repo, storage, manager, "Replacement policy.", replaces=first["policy_id"])
    staged = repo.get_revision(manager, replacement["revision_id"])
    assert staged["base_revision_id"] == first["revision_id"]
    assert repo.process_one_job(storage)
    assert repo.get_job(manager, replacement["id"])["status"] == "succeeded"
    storage.put_file("tampered", io.BytesIO(b"changed"))
    bad = repo.enqueue_upload(manager, "Bad hash", "hr", "tampered", "x.txt", "0"*64, 7)
    assert repo.process_one_job(storage)
    assert "checksum" in repo.get_job(manager, bad["id"])["error"]
    assert repo.get_policy(reader, bad["policy_id"])["published_revision_id"] is None


def test_search_returns_matching_tail_excerpt(setup):
    repo, storage, _, _, manager, reader, _ = setup
    text = ("普通条款内容。" * 240) + "\nSPECIAL_HOTEL_LIMIT 888 yuan"
    job = upload(repo, storage, manager, text)
    assert repo.process_one_job(storage)
    result = repo.search(reader, "SPECIAL_HOTEL_LIMIT", limit=1)
    assert "SPECIAL_HOTEL_LIMIT" in result[0]["text"]
    assert len(result[0]["text"]) <= repo.settings.chunk_size


def test_invalid_partial_evidence_and_request_key_reuse(setup):
    repo, storage, _, _, manager, *_ = setup
    job = publish_upload(setup)
    ev = evidence(repo, manager, job["policy_id"])
    with pytest.raises(ConflictError):
        repo.create_proposal(manager, "change", dict(ev, text="500"), "800", "reason")
    repo.create_proposal(manager, "change", ev, "800", "reason", request_key="once")
    with pytest.raises(ConflictError):
        repo.create_proposal(manager, "change", ev, "900", "reason", request_key="once")


def test_index_configuration_drift_fails_closed(setup):
    repo, storage, embedder, _, manager, *_ = setup
    publish_upload(setup)
    for field, value in [("embedding_model", "different-same-dimension-model"), ("chunk_size", 700), ("embedding_dimension", 32)]:
        changed = Repository(replace(repo.settings, **{field: value}), embedder=embedder)
        try:
            with pytest.raises(ConflictError, match="configuration changed"):
                changed.initialize()
        finally:
            changed.engine.dispose()


def test_get_block_uses_canonical_metadata_and_revision_list_is_small(setup):
    repo, storage, _, _, manager, reader, outsider = setup
    job = publish_upload(setup)
    ev = evidence(repo, manager, job["policy_id"])
    block = repo.get_block(reader, ev["revision_id"], ev["block_id"])
    assert block["text"] == ev["text"]
    assert block["policy_id"] == job["policy_id"] and block["revision_number"] == 1
    assert all("blocks" not in row for row in repo.list_revisions(reader, job["policy_id"]))
    with pytest.raises(PermissionError):
        repo.get_block(outsider, ev["revision_id"], ev["block_id"])


def test_exhausted_crashed_job_closes_staged_revision(setup):
    repo, storage, _, _, manager, *_ = setup
    job = upload(repo, storage, manager)
    claimed = repo._claim_job()
    with repo.engine.begin() as conn:
        conn.execute(repo.jobs.update().where(repo.jobs.c.id == job["id"]).values(
            attempts=repo.settings.job_max_attempts, lease_until=time.time()-1))
    assert not repo.process_one_job(storage)
    assert repo.get_job(manager, job["id"])["status"] == "failed"
    assert repo.get_revision(manager, job["revision_id"])["status"] == "failed"
    with pytest.raises(ConflictError, match="retry limit"):
        repo.retry_job(manager, job["id"])
