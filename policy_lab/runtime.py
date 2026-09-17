"""Checkpointer lifecycle and one in-flight invocation per owned graph thread."""
from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager

from sqlalchemy import text


class ThreadBusyError(RuntimeError):
    pass


_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()
_graph_slots = threading.BoundedSemaphore(4)


@contextmanager
def thread_execution(repository, thread_id):
    # A graph keeps one PostgreSQL connection for its session advisory lock and
    # needs other connections for node queries. Reserve pool capacity up front.
    if not _graph_slots.acquire(blocking=False):
        raise ThreadBusyError("The assistant is busy; retry this request shortly")
    try:
        with _owned_thread_lock(repository, thread_id):
            yield
    finally:
        _graph_slots.release()


@contextmanager
def _owned_thread_lock(repository, thread_id):
    if repository.engine.dialect.name == "postgresql":
        key = int.from_bytes(hashlib.sha256(thread_id.encode()).digest()[:8], "big", signed=True)
        with repository.engine.connect() as connection:
            if not connection.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}).scalar():
                raise ThreadBusyError("This workflow is already running")
            try:
                yield
            finally:
                connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
    else:
        with _guard:
            lock = _locks.setdefault(thread_id, threading.Lock())
            if not lock.acquire(blocking=False):
                raise ThreadBusyError("This workflow is already running")
        try:
            yield
        finally:
            with _guard:
                lock.release()
                _locks.pop(thread_id, None)


def make_checkpointer(settings, stack):
    if settings.database_url.startswith("postgresql"):
        from langgraph.checkpoint.postgres import PostgresSaver
        url = settings.checkpoint_url or settings.database_url.replace("postgresql+psycopg://", "postgresql://")
        saver = stack.enter_context(PostgresSaver.from_conn_string(url))
        saver.setup()
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        saver = stack.enter_context(SqliteSaver.from_conn_string(str(settings.data_dir / "checkpoints.db")))
        saver.setup()
    return saver
