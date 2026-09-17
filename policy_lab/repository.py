"""Durable policy revisions, access control, hybrid search and leased ingestion jobs.

PostgreSQL is the deployment backend. SQLite is a portable development/test adapter;
its Python search path is intentionally not the production retrieval implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid

import sqlalchemy as sa
from sqlalchemy import Column as C
from sqlalchemy.exc import IntegrityError

from .indexing import SentenceEmbedder, Reranker, cosine, digest, lexical_tokens, make_chunks


class ConflictError(ValueError):
    """The reviewed base changed, or this operation already exists."""


def _id():
    return uuid.uuid4().hex


def _dict(row):
    return dict(row._mapping) if row is not None else None


class Repository:
    def __init__(self, settings, embedder=None):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        kw = {"pool_pre_ping": True}
        if settings.database_url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if settings.database_url in {"sqlite://", "sqlite:///:memory:"}:
                from sqlalchemy.pool import StaticPool
                kw["poolclass"] = StaticPool
        self.engine = sa.create_engine(settings.database_url, **kw)
        self.pg = self.engine.dialect.name == "postgresql"
        self.embedder = embedder or SentenceEmbedder(settings)
        self.reranker = Reranker(settings) if settings.enable_reranker else None
        self.meta = sa.MetaData()
        def table(name, *cols, **kwargs):
            return sa.Table("policy_" + name, self.meta, *cols, **kwargs)
        def pk():
            return C("id", sa.String(64), primary_key=True)
        self.users = table("users", pk(), C("username", sa.String(200), unique=True, nullable=False),
            C("password_hash", sa.Text, nullable=False), C("role", sa.String(20), nullable=False),
            C("scopes", sa.JSON, nullable=False), C("active", sa.Boolean, nullable=False), C("created_at", sa.Float))
        self.sessions = table("sessions", C("token_hash", sa.String(128), primary_key=True),
            C("user_id", sa.String(64), nullable=False), C("expires_at", sa.Float, nullable=False))
        self.policies = table("documents", pk(), C("title", sa.Text, nullable=False),
            C("scope", sa.String(100), nullable=False, index=True), C("published_revision_id", sa.String(64)),
            C("created_by", sa.String(64)), C("created_at", sa.Float))
        self.revisions = table("revisions", pk(), C("policy_id", sa.String(64), nullable=False, index=True),
            C("number", sa.Integer, nullable=False), C("base_revision_id", sa.String(64)),
            C("status", sa.String(30), nullable=False), C("blocks", sa.JSON, nullable=False),
            C("original_key", sa.Text), C("original_name", sa.Text), C("sha256", sa.String(64)),
            C("created_by", sa.String(64)), C("created_at", sa.Float), C("published_at", sa.Float),
            sa.UniqueConstraint("policy_id", "number"))
        self.blocks = table("blocks", C("revision_id", sa.String(64), primary_key=True),
            C("id", sa.String(64), primary_key=True), C("ordinal", sa.Integer),
            C("heading", sa.Text), C("text", sa.Text), C("kind", sa.String(50)), C("locator", sa.JSON))
        self.jobs = table("jobs", pk(), C("actor_id", sa.String(64), nullable=False),
            C("kind", sa.String(30), nullable=False), C("status", sa.String(30), nullable=False, index=True),
            C("stage", sa.String(40)), C("progress", sa.Integer), C("error", sa.Text),
            C("policy_id", sa.String(64), nullable=False), C("revision_id", sa.String(64), nullable=False),
            C("proposal_id", sa.String(64)), C("attempts", sa.Integer, nullable=False),
            C("lease_token", sa.String(64)), C("lease_until", sa.Float), C("created_at", sa.Float), C("updated_at", sa.Float))
        self.proposals = table("proposals", pk(), C("actor_id", sa.String(64), nullable=False),
            C("policy_id", sa.String(64), nullable=False), C("base_revision_id", sa.String(64), nullable=False),
            C("block_id", sa.String(64), nullable=False), C("before_hash", sa.String(64)),
            C("before", sa.Text), C("after", sa.Text), C("instruction", sa.Text), C("reason", sa.Text),
            C("status", sa.String(30)), C("request_key", sa.String(200)), C("payload_hash", sa.String(64)),
            C("job_id", sa.String(64)), C("created_at", sa.Float), sa.UniqueConstraint("actor_id", "request_key"))
        self.audit = table("audit", pk(), C("actor_id", sa.String(64)), C("scope", sa.String(100)),
            C("policy_id", sa.String(64)), C("event", sa.String(80)), C("details", sa.JSON), C("created_at", sa.Float))
        self.threads = table("threads", C("id", sa.String(200), primary_key=True),
            C("actor_id", sa.String(64), nullable=False), C("created_at", sa.Float))
        vector_type = sa.JSON
        if self.pg:
            from pgvector.sqlalchemy import Vector
            vector_type = Vector(settings.embedding_dimension)
        self.chunks = table("chunks", pk(), C("policy_id", sa.String(64), index=True),
            C("revision_id", sa.String(64), index=True), C("block_id", sa.String(64)),
            C("text", sa.Text), C("lexical", sa.Text), C("vector", vector_type, nullable=False))
        self.cache = table("embedding_cache", C("id", sa.String(64), primary_key=True),
            C("vector", sa.JSON, nullable=False), C("created_at", sa.Float))
        self.configuration = table("index_configuration", C("id", sa.String(50), primary_key=True),
            C("configuration", sa.JSON, nullable=False), C("created_at", sa.Float))

    def initialize(self):
        with self._tx() as conn:
            if self.pg:
                conn.execute(sa.text("SELECT pg_advisory_xact_lock(817321)"))
                conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.meta.create_all(conn)
            configuration = {"schema": 1, "embedding_model": self.settings.embedding_model,
                "embedding_dimension": self.settings.embedding_dimension,
                "chunk_size": self.settings.chunk_size, "chunk_overlap": self.settings.chunk_overlap,
                "lexical_preprocessor": "cjk-unigram-bigram-v1", "context_format": "title-heading-text-v1"}
            persisted = conn.execute(sa.select(self.configuration.c.configuration).where(self.configuration.c.id == "active")).scalar()
            has_chunks = conn.execute(sa.select(self.chunks.c.id).limit(1)).first() is not None
            if (persisted is not None and persisted != configuration) or (has_chunks and persisted is None):
                raise ConflictError("Index configuration changed or is missing. Restore the previous model/chunk settings, or rebuild into a fresh database before serving queries.")
            if persisted is None:
                conn.execute(self.configuration.insert().values(id="active", configuration=configuration, created_at=time.time()))
            if self.pg:
                conn.execute(sa.text("CREATE INDEX IF NOT EXISTS policy_chunks_fts ON policy_chunks USING gin (to_tsvector('simple', lexical))"))
                conn.execute(sa.text("CREATE INDEX IF NOT EXISTS policy_chunks_vector ON policy_chunks USING hnsw (vector vector_cosine_ops)"))

    @contextmanager
    def _tx(self):
        with self.engine.connect() as conn:
            if not self.pg:
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                conn.begin()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _actor(self, conn, actor, *, write=False, scope=None, lock=False):
        user_id = actor.get("id") if isinstance(actor, dict) else actor
        statement = sa.select(self.users).where(self.users.c.id == user_id)
        if lock and self.pg:
            statement = statement.with_for_update()
        user = _dict(conn.execute(statement).first())
        if not user or not user["active"]:
            raise PermissionError("Account is inactive or unavailable")
        if write and user["role"] not in {"manager", "admin"}:
            raise PermissionError("Policy manager permission required")
        if scope is not None and user["role"] != "admin" and scope not in user["scopes"]:
            raise PermissionError("Policy scope access denied")
        return user

    def _policy(self, conn, actor, policy_id, write=False, lock=False):
        stmt = sa.select(self.policies).where(self.policies.c.id == policy_id)
        if lock and self.pg:
            stmt = stmt.with_for_update()
        policy = _dict(conn.execute(stmt).first())
        if not policy:
            raise KeyError("Policy not found")
        self._actor(conn, actor, write=write, scope=policy["scope"])
        return policy

    def _event(self, conn, actor_id, scope, policy_id, event, details):
        conn.execute(self.audit.insert().values(id=_id(), actor_id=actor_id, scope=scope,
            policy_id=policy_id, event=event, details=details, created_at=time.time()))

    @staticmethod
    def _user_values(role, scopes):
        if role not in {"reader", "manager", "admin"}:
            raise ValueError("Invalid role")
        if not isinstance(scopes, list) or len(scopes) > 100 or any(not isinstance(s, str) or not s.strip() or len(s) > 100 for s in scopes):
            raise ValueError("Invalid scopes")
        return sorted(set(s.strip() for s in scopes))

    def create_user(self, username, password_hash, role, scopes):
        scopes = self._user_values(role, scopes)
        if not isinstance(username, str) or not username.strip() or len(username) > 200:
            raise ValueError("Invalid username")
        row = dict(id=_id(), username=username.strip(), password_hash=password_hash,
            role=role, scopes=scopes, active=True, created_at=time.time())
        try:
            with self._tx() as conn:
                conn.execute(self.users.insert().values(**row))
        except IntegrityError:
            raise ConflictError("Username already exists") from None
        return row

    def get_user(self, user_id):
        with self.engine.connect() as conn:
            return _dict(conn.execute(sa.select(self.users).where(self.users.c.id == user_id)).first())

    def find_user(self, username):
        with self.engine.connect() as conn:
            return _dict(conn.execute(sa.select(self.users).where(self.users.c.username == username)).first())

    def list_users(self):
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(sa.select(self.users).order_by(self.users.c.created_at)).mappings()]

    def update_user(self, user_id, role, scopes, active):
        scopes = self._user_values(role, scopes)
        with self._tx() as conn:
            result = conn.execute(self.users.update().where(self.users.c.id == user_id).values(role=role, scopes=scopes, active=bool(active)))
            if not result.rowcount:
                raise KeyError("User not found")
        return self.get_user(user_id)

    def create_session(self, token_hash, user_id, expires_at):
        with self._tx() as conn:
            self._actor(conn, user_id)
            conn.execute(self.sessions.insert().values(token_hash=token_hash, user_id=user_id, expires_at=expires_at))

    def get_session(self, token_hash):
        with self.engine.connect() as conn:
            user_id = conn.execute(sa.select(self.sessions.c.user_id).where(
                self.sessions.c.token_hash == token_hash, self.sessions.c.expires_at > time.time())).scalar()
            if not user_id:
                return None
            try:
                return self._actor(conn, user_id)
            except PermissionError:
                return None

    def delete_session(self, token_hash):
        with self._tx() as conn:
            conn.execute(self.sessions.delete().where(self.sessions.c.token_hash == token_hash))

    def list_policies(self, actor, limit=50, offset=0):
        with self.engine.connect() as conn:
            actor = self._actor(conn, actor)
            stmt = sa.select(self.policies)
            if actor["role"] != "admin":
                stmt = stmt.where(self.policies.c.scope.in_(actor["scopes"]))
            return [dict(r) for r in conn.execute(stmt.order_by(self.policies.c.created_at.desc()).limit(min(max(limit, 1), 200)).offset(max(offset, 0))).mappings()]

    def get_policy(self, actor, policy_id):
        with self.engine.connect() as conn:
            return self._policy(conn, actor, policy_id)

    def list_revisions(self, actor, policy_id):
        with self.engine.connect() as conn:
            policy = self._policy(conn, actor, policy_id)
            user = self._actor(conn, actor)
            stmt = sa.select(*[column for column in self.revisions.c if column.name != "blocks"]).where(self.revisions.c.policy_id == policy_id)
            if user["role"] == "reader":
                stmt = stmt.where(self.revisions.c.status.in_(["published", "superseded"]))
            return [dict(r) for r in conn.execute(stmt.order_by(self.revisions.c.number.desc())).mappings()]

    def get_revision(self, actor, revision_id):
        with self.engine.connect() as conn:
            row = _dict(conn.execute(sa.select(self.revisions).where(self.revisions.c.id == revision_id)).first())
            if not row:
                raise KeyError("Revision not found")
            self._policy(conn, actor, row["policy_id"])
            user = self._actor(conn, actor)
            if user["role"] == "reader" and row["status"] not in {"published", "superseded"}:
                raise PermissionError("Unpublished revision")
            return row

    def get_block(self, actor, revision_id, block_id):
        with self.engine.connect() as conn:
            revision = _dict(conn.execute(sa.select(self.revisions.c.id, self.revisions.c.policy_id,
                self.revisions.c.number, self.revisions.c.status).where(self.revisions.c.id == revision_id)).first())
            if not revision:
                raise KeyError("Revision not found")
            self._policy(conn, actor, revision["policy_id"])
            user = self._actor(conn, actor)
            if user["role"] == "reader" and revision["status"] not in {"published", "superseded"}:
                raise PermissionError("Unpublished revision")
            block = _dict(conn.execute(sa.select(self.blocks).where(self.blocks.c.revision_id == revision_id, self.blocks.c.id == block_id)).first())
            if not block:
                raise KeyError("Clause not found")
            return dict(block, policy_id=revision["policy_id"], revision_number=revision["number"])

    def _store_blocks(self, conn, revision_id, blocks):
        conn.execute(self.blocks.delete().where(self.blocks.c.revision_id == revision_id))
        rows = [dict(revision_id=revision_id, **{key: block[key] for key in ("id", "ordinal", "heading", "text", "kind", "locator")}) for block in blocks]
        for offset in range(0, len(rows), 200):
            conn.execute(self.blocks.insert(), rows[offset:offset+200])

    def _revision(self, conn, policy, actor_id, *, blocks, original_key, original_name, sha256=None):
        number = (conn.execute(sa.select(sa.func.max(self.revisions.c.number)).where(self.revisions.c.policy_id == policy["id"])).scalar() or 0) + 1
        row = dict(id=_id(), policy_id=policy["id"], number=number,
            base_revision_id=policy["published_revision_id"], status="staged", blocks=blocks,
            original_key=original_key, original_name=original_name, sha256=sha256,
            created_by=actor_id, created_at=time.time(), published_at=None)
        conn.execute(self.revisions.insert().values(**row))
        self._store_blocks(conn, row["id"], blocks)
        return row

    def _job(self, conn, actor_id, kind, policy_id, revision_id, proposal_id=None):
        row = dict(id=_id(), actor_id=actor_id, kind=kind, status="queued", stage="queued", progress=0,
            error=None, policy_id=policy_id, revision_id=revision_id, proposal_id=proposal_id,
            attempts=0, lease_token=None, lease_until=None, created_at=time.time(), updated_at=time.time())
        conn.execute(self.jobs.insert().values(**row))
        return row

    def enqueue_upload(self, actor, title, scope, key, name, sha256, size, replaces=None):
        if not isinstance(title, str) or not title.strip() or len(title) > 500:
            raise ValueError("Title is required and must be at most 500 characters")
        if not isinstance(scope, str) or not scope or len(scope) > 100:
            raise ValueError("Invalid policy scope")
        if size <= 0 or size > self.settings.max_upload_bytes:
            raise ValueError("File exceeds the configured upload limit or is empty")
        with self._tx() as conn:
            user = self._actor(conn, actor, write=True, scope=scope)
            if self.pg:
                conn.execute(sa.text("SELECT pg_advisory_xact_lock(817322)"))
            if replaces:
                policy = self._policy(conn, user, replaces, write=True, lock=True)
                if policy["scope"] != scope:
                    raise ValueError("Replacement must retain the policy scope")
            else:
                if conn.execute(sa.select(sa.func.count()).select_from(self.policies)).scalar() >= self.settings.max_documents:
                    raise ValueError("Document capacity reached")
                policy = dict(id=_id(), title=title.strip(), scope=scope, published_revision_id=None,
                              created_by=user["id"], created_at=time.time())
            duplicate = conn.execute(sa.select(self.revisions.c.id).join(self.policies, self.revisions.c.policy_id == self.policies.c.id).where(
                self.revisions.c.sha256 == sha256, self.policies.c.scope == scope)).first()
            if duplicate:
                raise ConflictError("This file is already uploaded in this scope")
            if not replaces:
                conn.execute(self.policies.insert().values(**policy))
            revision = self._revision(conn, policy, user["id"], blocks=[], original_key=key,
                                      original_name=name, sha256=sha256)
            job = self._job(conn, user["id"], "upload", policy["id"], revision["id"])
            self._event(conn, user["id"], scope, policy["id"], "upload_queued", {"job_id": job["id"]})
            return job

    def _get_job(self, conn, actor, job_id):
        row = _dict(conn.execute(sa.select(self.jobs).where(self.jobs.c.id == job_id)).first())
        if not row:
            raise KeyError("Job not found")
        user = self._actor(conn, actor)
        policy = self._policy(conn, user, row["policy_id"])
        if user["role"] != "admin" and user["id"] != row["actor_id"]:
            raise PermissionError("Job belongs to another user")
        original_name = conn.execute(sa.select(self.revisions.c.original_name).where(self.revisions.c.id == row["revision_id"])).scalar()
        return dict(row, title=policy["title"], policy_title=policy["title"], original_name=original_name)

    def get_job(self, actor, job_id):
        with self.engine.connect() as conn:
            return self._get_job(conn, actor, job_id)

    def list_jobs(self, actor, limit=50):
        with self.engine.connect() as conn:
            user = self._actor(conn, actor)
            stmt = sa.select(self.jobs, self.policies.c.title.label("policy_title"), self.revisions.c.original_name).join(
                self.policies, self.jobs.c.policy_id == self.policies.c.id).join(self.revisions, self.jobs.c.revision_id == self.revisions.c.id)
            if user["role"] != "admin":
                stmt = stmt.where(self.jobs.c.actor_id == user["id"], self.policies.c.scope.in_(user["scopes"]))
            return [dict(r) for r in conn.execute(stmt.order_by(self.jobs.c.created_at.desc()).limit(min(max(limit, 1), 200))).mappings()]

    def retry_job(self, actor, job_id):
        with self._tx() as conn:
            job = self._get_job(conn, actor, job_id)
            self._policy(conn, actor, job["policy_id"], write=True)
            if job["status"] != "failed":
                raise ConflictError("Only failed jobs may be retried")
            if job["attempts"] >= self.settings.job_max_attempts:
                raise ConflictError("Job retry limit reached; upload a corrected document")
            conn.execute(self.jobs.update().where(self.jobs.c.id == job_id).values(status="queued", stage="queued", error=None, updated_at=time.time()))
            conn.execute(self.revisions.update().where(self.revisions.c.id == job["revision_id"], self.revisions.c.status == "failed").values(status="staged"))
        return self.get_job(actor, job_id)

    def snapshot(self, actor):
        with self.engine.connect() as conn:
            user = self._actor(conn, actor)
            stmt = sa.select(self.policies.c.id, self.policies.c.published_revision_id).where(self.policies.c.published_revision_id.is_not(None))
            if user["role"] != "admin":
                stmt = stmt.where(self.policies.c.scope.in_(user["scopes"]))
            return dict(conn.execute(stmt).all())

    def _query_vector(self, query):
        result = self.embedder.encode([query])[0]
        vector = [float(x) for x in result]
        if len(vector) != self.settings.embedding_dimension or not all(math.isfinite(x) for x in vector):
            raise ValueError("Embedding model dimension/value mismatch")
        return vector

    def search(self, actor, query, limit=12, editable=False, snapshot=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 12000:
            raise ValueError("Query must contain 1–12000 characters")
        with self.engine.connect() as conn:
            user = self._actor(conn, actor, write=editable)
        query_vector = self._query_vector(query)
        with self.engine.connect() as conn:
            user = self._actor(conn, user, write=editable)
            ch, doc, rev = self.chunks, self.policies, self.revisions
            source = ch.join(doc, ch.c.policy_id == doc.c.id).join(rev, ch.c.revision_id == rev.c.id)
            conditions = [rev.c.status.in_(["published", "superseded"])]
            if user["role"] != "admin":
                conditions.append(doc.c.scope.in_(user["scopes"]))
            if snapshot is None:
                conditions.append(ch.c.revision_id == doc.c.published_revision_id)
            else:
                if not snapshot:
                    return []
                # Each revision is checked against its own policy; untrusted snapshots cannot expose staged content.
                conditions.append(sa.tuple_(ch.c.policy_id, ch.c.revision_id).in_(list(snapshot.items())))
            candidates = max(1, min(self.settings.search_candidates, 100))
            select = sa.select(ch.c.id, ch.c.revision_id, ch.c.block_id, ch.c.text).select_from(source).where(*conditions)
            if self.pg:
                distance = ch.c.vector.cosine_distance(query_vector)
                dense = [dict(r) for r in conn.execute(select.order_by(distance).limit(candidates)).mappings()]
                terms = list(dict.fromkeys(lexical_tokens(query)))[:128]
                if terms:
                    ts_query = sa.func.to_tsquery(sa.literal_column("'simple'::regconfig"), " | ".join(terms))
                    ts_doc = sa.func.to_tsvector(sa.literal_column("'simple'"), ch.c.lexical)
                    lexical = [dict(r) for r in conn.execute(select.where(ts_doc.op("@@")(ts_query)).order_by(sa.func.ts_rank_cd(ts_doc, ts_query).desc()).limit(candidates)).mappings()]
                else:
                    lexical = []
            else:
                # Test adapter only. Production ranks and bounds candidates inside PostgreSQL.
                rows = [dict(r) for r in conn.execute(select.add_columns(ch.c.vector, ch.c.lexical)).mappings()]
                dense = sorted(rows, key=lambda r: cosine(r["vector"], query_vector), reverse=True)[:candidates]
                terms = set(lexical_tokens(query))
                lexical = sorted((r for r in rows if terms.intersection(r["lexical"].split())),
                    key=lambda r: len(terms.intersection(r["lexical"].split())), reverse=True)[:candidates]
            scores, hits = {}, {}
            for ranking in (dense, lexical):
                for rank, row in enumerate(ranking, 1):
                    scores[row["id"]] = scores.get(row["id"], 0) + 1 / (60 + rank)
                    hits[row["id"]] = row
            result, seen = [], set()
            for chunk_id in sorted(scores, key=scores.get, reverse=True):
                hit = hits[chunk_id]
                identity = (hit["revision_id"], hit["block_id"])
                if identity in seen:
                    continue
                seen.add(identity)
                revision = _dict(conn.execute(sa.select(rev.c.id, rev.c.policy_id, rev.c.number, doc.c.title, doc.c.scope).join(doc, rev.c.policy_id == doc.c.id).where(rev.c.id == hit["revision_id"])).first())
                block = _dict(conn.execute(sa.select(self.blocks.c.id, self.blocks.c.heading, self.blocks.c.locator).where(
                    self.blocks.c.revision_id == hit["revision_id"], self.blocks.c.id == hit["block_id"])).first())
                result.append(dict(id=":".join(identity), policy_id=revision["policy_id"], revision_id=revision["id"],
                    revision_number=revision["number"], block_id=block["id"], title=revision["title"], scope=revision["scope"],
                    heading=block.get("heading", ""), text=hit["text"], locator=block.get("locator", {}), score=scores[chunk_id]))
                if len(result) >= candidates:
                    break
        if result and self.reranker:
            result = [dict(row, score=score) for row, score in self.reranker.rank(query, result)]
        # Permission changes during model work must take effect before evidence is returned.
        with self.engine.connect() as conn:
            user = self._actor(conn, user, write=editable)
            if user["role"] != "admin":
                result = [row for row in result if row["scope"] in user["scopes"]]
        return result[:min(max(limit, 1), 50)]

    def _proposal(self, conn, actor, proposal_id):
        row = _dict(conn.execute(sa.select(self.proposals).where(self.proposals.c.id == proposal_id)).first())
        if not row:
            raise KeyError("Proposal not found")
        user = self._actor(conn, actor, write=True)
        policy = self._policy(conn, user, row["policy_id"], write=True)
        if row["actor_id"] != user["id"] and user["role"] != "admin":
            raise PermissionError("Proposal belongs to another user")
        number = conn.execute(sa.select(self.revisions.c.number).where(self.revisions.c.id == row["base_revision_id"])).scalar()
        heading = conn.execute(sa.select(self.blocks.c.heading).where(self.blocks.c.revision_id == row["base_revision_id"], self.blocks.c.id == row["block_id"])).scalar()
        return dict(row, title=policy["title"], scope=policy["scope"], base_revision_number=number, heading=heading or "")

    def create_proposal(self, actor, instruction, evidence, after, reason, request_key=None):
        if not isinstance(after, str) or not after.strip() or len(after) > min(self.settings.max_characters, 40000):
            raise ValueError("Replacement text must contain 1–40000 characters")
        payload_hash = digest(json.dumps([instruction, evidence.get("revision_id"), evidence.get("block_id"), evidence.get("text"), after, reason], ensure_ascii=False))
        with self._tx() as conn:
            user = self._actor(conn, actor, write=True)
            if request_key:
                if not isinstance(request_key, str) or len(request_key) > 200:
                    raise ValueError("Invalid proposal request key")
                if self.pg:
                    lock_id = int.from_bytes(hashlib.sha256(("proposal:" + user["id"] + ":" + request_key).encode()).digest()[:8], "big", signed=True)
                    conn.execute(sa.text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": lock_id})
                previous = _dict(conn.execute(sa.select(self.proposals).where(self.proposals.c.actor_id == user["id"], self.proposals.c.request_key == request_key)).first())
                if previous:
                    if previous["payload_hash"] != payload_hash:
                        raise ConflictError("Request key already used with different proposal content")
                    return self._proposal(conn, user, previous["id"])
            policy = self._policy(conn, user, evidence["policy_id"], write=True, lock=True)
            if policy["published_revision_id"] != evidence["revision_id"]:
                raise ConflictError("Policy changed; locate the clause again")
            revision = _dict(conn.execute(sa.select(self.revisions.c.id, self.revisions.c.number).where(self.revisions.c.id == evidence["revision_id"])).first())
            block = _dict(conn.execute(sa.select(self.blocks).where(self.blocks.c.revision_id == evidence["revision_id"], self.blocks.c.id == evidence["block_id"])).first())
            if not block or block["text"] != evidence["text"]:
                raise ConflictError("Evidence does not match the complete canonical clause")
            if after == block["text"]:
                raise ValueError("Proposal does not change the clause")
            row = dict(id=_id(), actor_id=user["id"], policy_id=policy["id"], base_revision_id=revision["id"],
                block_id=block["id"], before_hash=digest(block["text"]), before=block["text"], after=after,
                instruction=instruction, reason=reason, status="pending", request_key=request_key,
                payload_hash=payload_hash, job_id=None, created_at=time.time())
            conn.execute(self.proposals.insert().values(**row))
            self._event(conn, user["id"], policy["scope"], policy["id"], "proposal_created", {"proposal_id": row["id"], "block_id": block["id"]})
            return dict(row, title=policy["title"], scope=policy["scope"], base_revision_number=revision["number"], heading=block.get("heading", ""))

    def get_proposal(self, actor, proposal_id):
        with self.engine.connect() as conn:
            return self._proposal(conn, actor, proposal_id)

    def publish_proposal(self, actor, proposal_id):
        with self._tx() as conn:
            proposal = self._proposal(conn, actor, proposal_id)
            policy = self._policy(conn, actor, proposal["policy_id"], write=True, lock=True)
            # Re-read after policy lock, including idempotent requests racing each other.
            proposal = self._proposal(conn, actor, proposal_id)
            if proposal["job_id"]:
                return self._get_job(conn, actor, proposal["job_id"])
            if proposal["status"] != "pending":
                raise ConflictError("Proposal is no longer pending")
            if policy["published_revision_id"] != proposal["base_revision_id"]:
                raise ConflictError("Policy changed since review; create a new proposal")
            base = _dict(conn.execute(sa.select(self.revisions).where(self.revisions.c.id == proposal["base_revision_id"])).first())
            blocks = copy.deepcopy(base["blocks"])
            block = next(b for b in blocks if b["id"] == proposal["block_id"])
            if digest(block["text"]) != proposal["before_hash"]:
                raise ConflictError("Original clause changed")
            block["text"] = proposal["after"]
            block["locator"] = dict(block.get("locator", {}), canonical_override=True,
                                    original_revision_id=base["id"])
            user = self._actor(conn, actor, write=True)
            revision = self._revision(conn, policy, user["id"], blocks=blocks,
                original_key=base["original_key"], original_name=base["original_name"])
            job = self._job(conn, user["id"], "proposal", policy["id"], revision["id"], proposal_id)
            conn.execute(self.proposals.update().where(self.proposals.c.id == proposal_id).values(status="queued", job_id=job["id"]))
            self._event(conn, user["id"], policy["scope"], policy["id"], "proposal_approved", {"proposal_id": proposal_id, "job_id": job["id"]})
            return job

    def reject_proposal(self, actor, proposal_id):
        with self._tx() as conn:
            proposal = self._proposal(conn, actor, proposal_id)
            if proposal["status"] == "rejected":
                return proposal
            if proposal["status"] != "pending":
                raise ConflictError("Proposal is no longer pending")
            conn.execute(self.proposals.update().where(self.proposals.c.id == proposal_id).values(status="rejected"))
            self._event(conn, actor["id"], proposal["scope"], proposal["policy_id"], "proposal_rejected", {"proposal_id": proposal_id})
        return self.get_proposal(actor, proposal_id)

    def rollback(self, actor, policy_id, revision_id, base_revision_id):
        with self._tx() as conn:
            policy = self._policy(conn, actor, policy_id, write=True, lock=True)
            if policy["published_revision_id"] != base_revision_id:
                raise ConflictError("Policy changed since rollback review")
            target = _dict(conn.execute(sa.select(self.revisions).where(self.revisions.c.id == revision_id,
                self.revisions.c.policy_id == policy_id, self.revisions.c.status.in_(["published", "superseded"]))).first())
            if not target:
                raise ValueError("Rollback target must be a published revision of this policy")
            if revision_id == base_revision_id:
                raise ValueError("This revision is already current")
            revision = self._revision(conn, policy, actor["id"], blocks=copy.deepcopy(target["blocks"]),
                original_key=target["original_key"], original_name=target["original_name"])
            job = self._job(conn, actor["id"], "rollback", policy_id, revision["id"])
            self._event(conn, actor["id"], policy["scope"], policy_id, "rollback_queued", {"target_revision_id": revision_id, "job_id": job["id"]})
            return job

    def audit_events(self, actor, limit=100):
        with self.engine.connect() as conn:
            user = self._actor(conn, actor)
            stmt = sa.select(self.audit, self.users.c.username.label("actor_username"), self.policies.c.title.label("policy_title")).select_from(
                self.audit.outerjoin(self.users, self.audit.c.actor_id == self.users.c.id).outerjoin(self.policies, self.audit.c.policy_id == self.policies.c.id))
            if user["role"] != "admin":
                stmt = stmt.where(self.audit.c.scope.in_(user["scopes"]))
            return [dict(r) for r in conn.execute(stmt.order_by(self.audit.c.created_at.desc()).limit(min(max(limit, 1), 500))).mappings()]

    def create_thread(self, actor, thread_id):
        with self._tx() as conn:
            user = self._actor(conn, actor)
            existing = conn.execute(sa.select(self.threads.c.actor_id).where(self.threads.c.id == thread_id)).scalar()
            if existing:
                if existing != user["id"]:
                    raise PermissionError("Conversation belongs to another user")
                return
            conn.execute(self.threads.insert().values(id=thread_id, actor_id=user["id"], created_at=time.time()))

    def authorize_thread(self, actor, thread_id):
        with self.engine.connect() as conn:
            user = self._actor(conn, actor)
            owner = conn.execute(sa.select(self.threads.c.actor_id).where(self.threads.c.id == thread_id)).scalar()
            if owner != user["id"]:
                raise PermissionError("Conversation belongs to another user or is unavailable")
            return True

    def _claim_job(self):
        now = time.time()
        with self._tx() as conn:
            exhausted = sa.and_(self.jobs.c.status == "processing", self.jobs.c.lease_until < now,
                               self.jobs.c.attempts >= self.settings.job_max_attempts)
            exhausted_revisions = list(conn.execute(sa.select(self.jobs.c.revision_id).where(exhausted)).scalars())
            if exhausted_revisions:
                conn.execute(self.revisions.update().where(self.revisions.c.id.in_(exhausted_revisions), self.revisions.c.status == "staged").values(status="failed"))
            conn.execute(self.jobs.update().where(self.jobs.c.status == "processing", self.jobs.c.lease_until < now,
                self.jobs.c.attempts >= self.settings.job_max_attempts).values(status="failed", stage="failed", error="Worker lease expired; retry limit reached", updated_at=now))
            stmt = sa.select(self.jobs).where(sa.or_(self.jobs.c.status == "queued", sa.and_(self.jobs.c.status == "processing", self.jobs.c.lease_until < now)),
                self.jobs.c.attempts < self.settings.job_max_attempts).order_by(self.jobs.c.created_at).limit(1)
            if self.pg:
                stmt = stmt.with_for_update(skip_locked=True)
            job = _dict(conn.execute(stmt).first())
            if not job:
                return None
            job.update(status="processing", stage="validating", progress=5, attempts=job["attempts"] + 1,
                       lease_token=_id(), lease_until=now + self.settings.job_lease_seconds, updated_at=now, error=None)
            conn.execute(self.jobs.update().where(self.jobs.c.id == job["id"]).values(**{k:v for k,v in job.items() if k != "id"}))
            return job

    def _progress(self, job, stage=None, progress=None):
        values = {"lease_until": time.time() + self.settings.job_lease_seconds, "updated_at": time.time()}
        if stage:
            values.update(stage=stage, progress=progress)
        with self._tx() as conn:
            result = conn.execute(self.jobs.update().where(self.jobs.c.id == job["id"], self.jobs.c.lease_token == job["lease_token"],
                self.jobs.c.status == "processing").values(**values))
            if not result.rowcount:
                raise ConflictError("Worker lease lost")

    def _check_lease(self, conn, job):
        stmt = sa.select(self.jobs).where(self.jobs.c.id == job["id"])
        if self.pg:
            stmt = stmt.with_for_update()
        lease = _dict(conn.execute(stmt).first())
        if not lease or lease["lease_token"] != job["lease_token"] or lease["status"] != "processing" or lease["lease_until"] < time.time():
            raise ConflictError("Worker lease lost before writing prepared content")

    def _index_revision(self, policy, revision, job):
        chunks = list(make_chunks(policy["title"], revision["blocks"], self.settings.chunk_size, self.settings.chunk_overlap))
        if not chunks:
            raise ValueError("Document contains no searchable text")
        fingerprint = f"v1:{self.settings.embedding_model}:{self.settings.embedding_dimension}:"
        indexed = []
        for offset in range(0, len(chunks), 32):
            batch = chunks[offset:offset + 32]
            keys = [digest(fingerprint + row["context"]) for row in batch]
            with self.engine.connect() as conn:
                cached = {r["id"]: r["vector"] for r in conn.execute(sa.select(self.cache).where(self.cache.c.id.in_(keys))).mappings()}
            missing = list(dict.fromkeys(key for key in keys if key not in cached))
            if missing:
                texts = {key: row["context"] for key, row in zip(keys, batch)}
                vectors = self.embedder.encode([texts[key] for key in missing])
                if len(vectors) != len(missing):
                    raise ValueError("Embedding model returned an incorrect batch size")
                with self._tx() as conn:
                    for key, raw_vector in zip(missing, vectors):
                        vector = [float(x) for x in raw_vector]
                        if len(vector) != self.settings.embedding_dimension or not all(math.isfinite(x) for x in vector):
                            raise ValueError("Embedding model dimension/value mismatch")
                        cached[key] = vector
                        if self.pg:
                            from sqlalchemy.dialects.postgresql import insert
                        else:
                            from sqlalchemy.dialects.sqlite import insert
                        conn.execute(insert(self.cache).values(id=key, vector=vector, created_at=time.time()).on_conflict_do_nothing(index_elements=["id"]))
            for key, chunk in zip(keys, batch):
                indexed.append(dict(id=_id(), policy_id=policy["id"], revision_id=revision["id"], block_id=chunk["block_id"],
                                    text=chunk["text"], lexical=chunk["lexical"], vector=cached[key]))
            self._progress(job, "embedding", 40 + int(45 * min(offset + 32, len(chunks)) / len(chunks)))
        with self._tx() as conn:
            self._check_lease(conn, job)
            conn.execute(self.chunks.delete().where(self.chunks.c.revision_id == revision["id"]))
            for offset in range(0, len(indexed), 200):
                conn.execute(self.chunks.insert(), indexed[offset:offset+200])

    def _commit_job(self, job, revision):
        with self._tx() as conn:
            self._check_lease(conn, job)
            self._actor(conn, job["actor_id"], write=True, lock=True)
            policy = self._policy(conn, job["actor_id"], job["policy_id"], write=True, lock=True)
            expected = revision["base_revision_id"]
            if policy["published_revision_id"] != expected:
                raise ConflictError("Another revision was published; this prepared revision was not activated")
            # Both indexed content and canonical version become visible through this pointer.
            predicate = self.policies.c.published_revision_id.is_(None) if expected is None else self.policies.c.published_revision_id == expected
            result = conn.execute(self.policies.update().where(self.policies.c.id == policy["id"], predicate).values(published_revision_id=revision["id"]))
            if result.rowcount != 1:
                raise ConflictError("Publication conflict")
            if expected:
                conn.execute(self.revisions.update().where(self.revisions.c.id == expected).values(status="superseded"))
            conn.execute(self.revisions.update().where(self.revisions.c.id == revision["id"]).values(status="published", published_at=time.time()))
            conn.execute(self.jobs.update().where(self.jobs.c.id == job["id"]).values(status="succeeded", stage="published", progress=100, error=None, lease_until=None, updated_at=time.time()))
            if job["proposal_id"]:
                conn.execute(self.proposals.update().where(self.proposals.c.id == job["proposal_id"]).values(status="published"))
            self._event(conn, job["actor_id"], policy["scope"], policy["id"], "revision_published", {"revision_id": revision["id"], "base_revision_id": expected, "kind": job["kind"], "job_id": job["id"]})

    def process_one_job(self, storage=None, parser=None):
        job = self._claim_job()
        if not job:
            return False
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(max(1, min(30, self.settings.job_lease_seconds / 3))):
                try:
                    self._progress(job)
                except Exception:
                    return
        thread = threading.Thread(target=heartbeat, name="policy-job-heartbeat", daemon=True)
        thread.start()
        try:
            with self.engine.connect() as conn:
                policy = self._policy(conn, job["actor_id"], job["policy_id"], write=True)
                revision = _dict(conn.execute(sa.select(self.revisions).where(self.revisions.c.id == job["revision_id"])).first())
            if job["kind"] == "upload":
                from .storage import make_storage
                from .parsing import DocumentParser
                storage = storage or make_storage(self.settings)
                parser = parser or DocumentParser(self.settings)
                self._progress(job, "parsing", 15)
                with tempfile.TemporaryDirectory(prefix="policy-parse-", dir=self.settings.data_dir) as directory:
                    path = Path(directory) / ("source" + Path(revision["original_name"]).suffix.lower())
                    hasher = hashlib.sha256()
                    size = 0
                    with storage.open(revision["original_key"]) as source, path.open("wb") as output:
                        while True:
                            data = source.read(1024*1024)
                            if not data:
                                break
                            size += len(data)
                            if size > self.settings.max_upload_bytes:
                                raise ValueError("Stored object exceeds upload limit")
                            hasher.update(data)
                            output.write(data)
                    if hasher.hexdigest() != revision["sha256"]:
                        raise ValueError("Stored object checksum mismatch")
                    blocks = parser.parse(path, revision["original_name"]) if hasattr(parser, "parse") else parser(path, revision["original_name"])
                from .parsing import validate_blocks
                revision["blocks"] = validate_blocks(blocks, self.settings)
                with self._tx() as conn:
                    self._check_lease(conn, job)
                    conn.execute(self.revisions.update().where(self.revisions.c.id == revision["id"]).values(blocks=revision["blocks"]))
                    self._store_blocks(conn, revision["id"], revision["blocks"])
            self._progress(job, "indexing", 40)
            self._index_revision(policy, revision, job)
            self._progress(job, "publishing", 95)
            self._commit_job(job, revision)
        except Exception as exc:
            with self._tx() as conn:
                result = conn.execute(self.jobs.update().where(self.jobs.c.id == job["id"], self.jobs.c.lease_token == job["lease_token"], self.jobs.c.status == "processing").values(
                    status="failed", stage="failed", error=f"{type(exc).__name__}: {str(exc)[:1200]}", lease_until=None, updated_at=time.time()))
                if result.rowcount:
                    conn.execute(self.revisions.update().where(self.revisions.c.id == job["revision_id"], self.revisions.c.status == "staged").values(status="failed"))
        finally:
            stop.set()
            thread.join(timeout=2)
        return True
