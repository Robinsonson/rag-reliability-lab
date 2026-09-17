"""Execution admission reserves DB connections and releases locks on failure."""
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from policy_lab.runtime import ThreadBusyError, thread_execution


def repository():
    return SimpleNamespace(engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))


def test_admission_reserves_connection_capacity_and_recovers_after_exit():
    repo = repository()
    with ExitStack() as stack:
        for i in range(4):
            stack.enter_context(thread_execution(repo, f"workflow-{i}"))
        with pytest.raises(ThreadBusyError, match="busy"):
            with thread_execution(repo, "fifth"):
                pytest.fail("Admission limit was bypassed")
    with thread_execution(repo, "after-release"):
        pass


def test_same_thread_is_exclusive_and_exception_releases_slot():
    repo = repository()
    with pytest.raises(RuntimeError, match="fixture"):
        with thread_execution(repo, "same-workflow"):
            with pytest.raises(ThreadBusyError, match="already running"):
                with thread_execution(repo, "same-workflow"):
                    pytest.fail("Thread lock was bypassed")
            raise RuntimeError("fixture")
    with thread_execution(repo, "same-workflow"):
        pass
