"""Workflow invariants against real LangGraph execution, without cloud calls."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import ValidationError

from policy_lab.config import Settings
from policy_lab.graphs import _apply_span_patch, build_graphs
from policy_lab.llm import ModelTransientError, ModelUnavailable, StructuredLLM, complete_structured


class ScriptedModel:
    def __init__(self, **responses):
        self.responses = {task: deque(items) for task, items in responses.items()}
        self.calls = defaultdict(list)

    def complete(self, task, payload):
        self.calls[task].append(deepcopy(payload))
        response = self.responses[task].popleft()
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(payload)
        return deepcopy(response)


class RepositoryDouble:
    def __init__(self):
        self.settings = Settings()
        self.users = {
            "manager": {"id": "manager", "username": "manager", "role": "manager", "scopes": ["hr"], "active": True},
            "other": {"id": "other", "username": "other", "role": "manager", "scopes": ["hr"], "active": True},
            "reader": {"id": "reader", "username": "reader", "role": "reader", "scopes": ["hr"], "active": True},
        }
        self.policy = {"id": "policy", "title": "差旅政策", "scope": "hr", "published_revision_id": "revision-1"}
        self.revision = {"id": "revision-1", "policy_id": "policy", "number": 1,
                         "blocks": [{"id": "clause", "heading": "国内住宿", "text": "国内出差住宿每晚限额为 600 元。", "locator": {"page": 3}}]}
        self.evidence = {"id": "revision-1:clause", "policy_id": "policy", "revision_id": "revision-1",
                         "revision_number": 1, "block_id": "clause", "title": "差旅政策", "scope": "hr",
                         "heading": "国内住宿", "text": self.revision["blocks"][0]["text"], "locator": {"page": 3}, "score": .9}
        self.threads = {"answer": "manager", "update": "manager", "reader": "reader"}
        self.searches = []
        self.search_results = None
        self.proposals = {}
        self.request_keys = {}
        self.publish_calls = 0
        self.jobs = {}
        self.crash_after_enqueue = False
        self.crash_after_proposal = False

    def get_user(self, user_id):
        return deepcopy(self.users.get(user_id))

    def authorize_thread(self, actor, thread_id):
        if self.threads.get(thread_id) != actor["id"]:
            raise PermissionError("Not the workflow owner")

    def _acl(self, actor):
        if self.policy["scope"] not in actor["scopes"]:
            raise PermissionError("Policy scope denied")

    def snapshot(self, actor):
        self._acl(actor)
        return {"policy": self.policy["published_revision_id"]}

    def search(self, actor, query, limit=12, editable=False, snapshot=None):
        self._acl(actor)
        self.searches.append({"actor": actor["id"], "query": query, "editable": editable, "snapshot": snapshot})
        return deepcopy(self.search_results if self.search_results is not None else [self.evidence])

    def get_policy(self, actor, policy_id):
        self._acl(actor)
        assert policy_id == "policy"
        return deepcopy(self.policy)

    def get_revision(self, actor, revision_id):
        self._acl(actor)
        assert revision_id == "revision-1"
        return deepcopy(self.revision)

    def create_proposal(self, actor, instruction, evidence, after, reason, request_key=None):
        self._acl(actor)
        key = (actor["id"], request_key)
        if key in self.request_keys:
            return deepcopy(self.proposals[self.request_keys[key]])
        assert evidence["text"] == self.revision["blocks"][0]["text"]
        proposal = {"id": "proposal-1", "actor_id": actor["id"], "policy_id": "policy", "title": self.policy["title"],
                    "scope": "hr", "base_revision_id": "revision-1", "block_id": "clause", "before": evidence["text"],
                    "after": after, "reason": reason, "status": "pending"}
        self.proposals[proposal["id"]] = proposal
        self.request_keys[key] = proposal["id"]
        if self.crash_after_proposal:
            self.crash_after_proposal = False
            raise ConnectionError("Simulated process loss after proposal commit")
        return deepcopy(proposal)

    def get_proposal(self, actor, proposal_id):
        self._acl(actor)
        proposal = self.proposals[proposal_id]
        if proposal["actor_id"] != actor["id"]:
            raise PermissionError("Not the proposal owner")
        return deepcopy(proposal)

    def publish_proposal(self, actor, proposal_id):
        self._acl(actor)
        proposal = self.get_proposal(actor, proposal_id)
        if proposal_id in self.jobs:
            return deepcopy(self.jobs[proposal_id])
        if proposal["base_revision_id"] != self.policy["published_revision_id"]:
            raise ValueError("Stale base revision")
        self.publish_calls += 1
        job = {"id": "job-1", "status": "queued", "policy_id": "policy"}
        self.jobs[proposal_id] = job
        if self.crash_after_enqueue:
            self.crash_after_enqueue = False
            raise ConnectionError("Simulated process loss after the database commit")
        return deepcopy(job)

    def reject_proposal(self, actor, proposal_id):
        self._acl(actor)
        self.proposals[proposal_id]["status"] = "rejected"


def config(thread="update"):
    return {"configurable": {"thread_id": thread}, "recursion_limit": 40}


def update_model(**extra):
    responses = {"select_clause": [{"selected_id": "revision-1:clause"}],
                 "draft_clause": [{"before_span": "600", "after_span": "800", "reason": "调整住宿限额"}]}
    responses.update(extra)
    return ScriptedModel(**responses)


def start_update(repo, model, checkpointer=None):
    graph = build_graphs(repo, checkpointer or InMemorySaver(), model)[1]
    paused = graph.invoke({"actor_id": "manager", "instruction": "把国内住宿限额改成 800 元"}, config())
    return graph, paused


def test_answer_is_grounded_in_original_metadata_and_cites_sources():
    repo = RepositoryDouble()
    model = ScriptedModel(assess_evidence=[{"sufficient": True, "relevant_ids": [repo.evidence["id"]]}],
                          answer=[{"answer": "国内住宿每晚限额为 600 元。[S1]", "citation_ids": ["S1"]}])
    graph = build_graphs(repo, InMemorySaver(), model)[0]
    result = graph.invoke({"actor_id": "manager", "question": "住宿限额是多少？"}, config("answer"))["result"]
    assert result["answerable"] is True
    assert result["citations"][0]["text"] == repo.evidence["text"]
    assert result["citations"][0]["locator"] == {"page": 3}
    assert result["citations"][0]["revision_id"] == "revision-1"
    assert result["verification"] == "citation_ids_and_source_metadata_only"
    assert repo.searches[0]["snapshot"] == {"policy": "revision-1"}
    assert repo.publish_calls == 0


def test_answer_retrieval_has_exactly_one_rewrite_budget():
    repo = RepositoryDouble()
    insufficient = {"sufficient": False, "relevant_ids": [], "rewritten_query": "international hotel policy"}
    model = ScriptedModel(assess_evidence=[insufficient, insufficient])
    graph = build_graphs(repo, InMemorySaver(), model)[0]
    result = graph.invoke({"actor_id": "manager", "question": "海外出差呢？"}, config("answer"))["result"]
    assert len(repo.searches) == 2
    assert result["answerable"] is False
    assert len(model.calls["answer"]) == 0


def test_retrieved_excerpt_at_end_of_large_block_is_preserved():
    repo = RepositoryDouble()
    excerpt = "海外住宿需要财务主管批准。"
    repo.revision["blocks"][0]["text"] = "无关的历史条款。" * 1000 + excerpt
    repo.evidence["text"] = excerpt
    model = ScriptedModel(assess_evidence=[{"sufficient": True, "relevant_ids": [repo.evidence["id"]]}],
                          answer=[{"answer": "需要财务主管批准。[S1]", "citation_ids": ["S1"]}])
    graph = build_graphs(repo, InMemorySaver(), model)[0]
    result = graph.invoke({"actor_id": "manager", "question": "海外住宿谁批准？"}, config("answer"))["result"]
    assert model.calls["assess_evidence"][0]["evidence"][0]["text"] == excerpt
    assert model.calls["answer"][0]["sources"][0]["text"] == excerpt
    assert result["citations"][0]["text"] == excerpt


def test_search_excerpt_must_be_an_actual_canonical_substring():
    repo = RepositoryDouble()
    repo.evidence["text"] = "Invented policy text that is not in the revision"
    graph = build_graphs(repo, InMemorySaver(), ScriptedModel())[0]
    with pytest.raises(ValueError, match="canonical"):
        graph.invoke({"actor_id": "manager", "question": "住宿限额？"}, config("answer"))


@pytest.mark.parametrize("text,labels", [("800 元 [S99]", ["S99"]), ("800 元", ["S1"]), ("800 元 [S1]", ["S2"])])
def test_fabricated_or_missing_citations_fail_closed(text, labels):
    repo = RepositoryDouble()
    model = ScriptedModel(assess_evidence=[{"sufficient": True, "relevant_ids": [repo.evidence["id"]]}],
                          answer=[{"answer": text, "citation_ids": labels}])
    graph = build_graphs(repo, InMemorySaver(), model)[0]
    result = graph.invoke({"actor_id": "manager", "question": "住宿限额？"}, config("answer"))["result"]
    assert result["answerable"] is False
    assert result["citations"] == []


def test_model_cannot_select_evidence_outside_search_results():
    repo = RepositoryDouble()
    model = ScriptedModel(assess_evidence=[{"sufficient": True, "relevant_ids": ["secret-payroll"]}])
    graph = build_graphs(repo, InMemorySaver(), model)[0]
    with pytest.raises(ValueError, match="unknown evidence"):
        graph.invoke({"actor_id": "manager", "question": "工资？"}, config("answer"))


def test_read_permission_is_rechecked_before_generation():
    repo = RepositoryDouble()

    def revoke(_):
        repo.users["manager"]["scopes"] = []
        return {"sufficient": True, "relevant_ids": [repo.evidence["id"]]}

    model = ScriptedModel(assess_evidence=[revoke])
    graph = build_graphs(repo, InMemorySaver(), model)[0]
    with pytest.raises(PermissionError):
        graph.invoke({"actor_id": "manager", "question": "住宿限额？"}, config("answer"))
    assert len(model.calls["answer"]) == 0


def test_empty_authorized_search_abstains_without_fabricated_answer():
    repo = RepositoryDouble()
    repo.search_results = []
    graph = build_graphs(repo, InMemorySaver(), ScriptedModel())[0]
    result = graph.invoke({"actor_id": "manager", "question": "股票期权？"}, config("answer"))["result"]
    assert result["answerable"] is False
    assert len(repo.searches) == 1


def test_review_persists_across_process_restart_then_queues_publication(tmp_path):
    repo = RepositoryDouble()
    model = update_model()
    db_path = str(tmp_path / "checkpoints.sqlite")
    with SqliteSaver.from_conn_string(db_path) as saver:
        _, paused = start_update(repo, model, saver)
        payload = paused["__interrupt__"][0].value
        assert payload["type"] == "review"
        assert payload["proposal"]["before"] == repo.evidence["text"]
        assert payload["proposal"]["after"] == "国内出差住宿每晚限额为 800 元。"
        assert repo.publish_calls == 0
    with SqliteSaver.from_conn_string(db_path) as saver:
        resumed_graph = build_graphs(repo, saver, model)[1]
        result = resumed_graph.invoke(Command(resume={"approve": True}), config())["result"]
    assert result["status"] == "queued"
    assert result["job"]["status"] == "queued"
    assert repo.policy["published_revision_id"] == "revision-1"
    assert repo.publish_calls == 1
    assert len(model.calls["draft_clause"]) == 1
    assert len(repo.proposals) == 1


def test_publication_resume_after_commit_is_idempotent():
    repo = RepositoryDouble()
    graph, _ = start_update(repo, update_model())
    repo.crash_after_enqueue = True
    with pytest.raises(ConnectionError):
        graph.invoke(Command(resume={"approve": True}), config())
    result = graph.invoke(None, config())["result"]
    assert result["status"] == "queued"
    assert repo.publish_calls == 1
    assert len(repo.jobs) == 1


def test_proposal_resume_after_commit_uses_same_request_key():
    repo = RepositoryDouble()
    repo.crash_after_proposal = True
    model = update_model()
    graph = build_graphs(repo, InMemorySaver(), model)[1]
    with pytest.raises(ConnectionError):
        graph.invoke({"actor_id": "manager", "instruction": "住宿限额改为 800 元"}, config())
    paused = graph.invoke(None, config())
    assert paused["__interrupt__"][0].value["type"] == "review"
    assert len(repo.proposals) == 1
    assert ("manager", "update") in repo.request_keys
    assert len(model.calls["draft_clause"]) == 1


def test_update_uses_full_canonical_block_even_when_search_returns_excerpt():
    repo = RepositoryDouble()
    repo.revision["blocks"][0]["text"] = "其他条款保持。国内出差住宿每晚限额为 600 元。例外须审批。"
    repo.evidence["text"] = "国内出差住宿每晚限额为 600 元。"
    model = update_model(draft_clause=[{"before_span": "600", "after_span": "800", "reason": "仅调整额度"}])
    _, paused = start_update(repo, model)
    assert model.calls["draft_clause"][0]["clause"]["text"] == repo.revision["blocks"][0]["text"]
    assert paused["__interrupt__"][0].value["proposal"]["before"].endswith("例外须审批。")


@pytest.mark.parametrize("change", ["reader", "inactive", "scope"])
def test_review_resume_rechecks_current_account_and_permissions(change):
    repo = RepositoryDouble()
    graph, _ = start_update(repo, update_model())
    if change == "reader":
        repo.users["manager"]["role"] = "reader"
    elif change == "inactive":
        repo.users["manager"]["active"] = False
    else:
        repo.users["manager"]["scopes"] = []
    with pytest.raises(PermissionError):
        graph.invoke(Command(resume={"approve": True}), config())
    assert repo.publish_calls == 0


def test_thread_owner_check_is_independent_of_role():
    repo = RepositoryDouble()
    graph = build_graphs(repo, InMemorySaver(), update_model())[1]
    with pytest.raises(PermissionError, match="owner"):
        graph.invoke({"actor_id": "other", "instruction": "调整住宿"}, config())


def test_reader_cannot_start_policy_change():
    repo = RepositoryDouble()
    graph = build_graphs(repo, InMemorySaver(), update_model())[1]
    with pytest.raises(PermissionError, match="manager"):
        graph.invoke({"actor_id": "reader", "instruction": "调整住宿"}, config("reader"))


def test_stale_revision_cannot_be_approved():
    repo = RepositoryDouble()
    graph, _ = start_update(repo, update_model())
    repo.policy["published_revision_id"] = "revision-2"
    with pytest.raises(ValueError, match="Stale"):
        graph.invoke(Command(resume={"approve": True}), config())
    assert repo.publish_calls == 0


def test_rejection_and_boolean_review_are_explicit():
    repo = RepositoryDouble()
    graph, _ = start_update(repo, update_model())
    invalid = graph.invoke(Command(resume={"approve": "true"}), config())
    assert "explicit" in invalid["__interrupt__"][0].value["error"]
    result = graph.invoke(Command(resume={"approve": False}), config())["result"]
    assert result["status"] == "rejected"
    assert repo.proposals["proposal-1"]["status"] == "rejected"
    assert repo.publish_calls == 0


def test_business_clarification_selects_a_system_provided_candidate():
    repo = RepositoryDouble()
    model = update_model(select_clause=[{"selected_id": None, "question": "调整国内还是海外的住宿标准？"}])
    graph, paused = start_update(repo, model)
    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "clarification"
    assert payload["candidates"][0]["title"] == "差旅政策"
    invalid = graph.invoke(Command(resume={"choice_id": "not-a-candidate"}), config())
    assert "displayed" in invalid["__interrupt__"][0].value["error"]
    reviewed = graph.invoke(Command(resume={"choice_id": payload["candidates"][0]["id"]}), config())
    assert reviewed["__interrupt__"][0].value["type"] == "review"
    assert repo.publish_calls == 0


def test_freeform_clarification_is_bounded_to_two_rounds():
    repo = RepositoryDouble()
    ambiguous = {"selected_id": None, "question": "适用哪个地区？"}
    model = ScriptedModel(select_clause=[ambiguous, ambiguous, ambiguous])
    graph, paused = start_update(repo, model)
    assert paused["__interrupt__"]
    paused = graph.invoke(Command(resume={"clarification": "国内"}), config())
    assert paused["__interrupt__"]
    done = graph.invoke(Command(resume={"clarification": "上海"}), config())
    assert done["result"]["status"] == "needs_clarification"
    assert not done.get("__interrupt__")
    assert not repo.proposals


def test_model_transient_errors_have_bounded_graph_retry():
    repo = RepositoryDouble()
    model = update_model(select_clause=[ModelTransientError("try again"), {"selected_id": "revision-1:clause"}])
    _, paused = start_update(repo, model)
    assert paused["__interrupt__"][0].value["type"] == "review"
    assert len(model.calls["select_clause"]) == 2


def test_unconfigured_model_fails_without_a_fake_answer():
    model = StructuredLLM(replace(Settings(), chat_api_key=""))
    with pytest.raises(ModelUnavailable, match="OPENAI_API_KEY"):
        model.complete("answer", {})


def test_injected_model_cannot_add_authorization_or_write_fields():
    model = ScriptedModel(select_clause=[{"selected_id": "clause", "approve": True}])
    with pytest.raises(ValidationError):
        complete_structured(model, "select_clause", {})


def test_large_canonical_block_is_updated_with_a_short_model_patch():
    repo = RepositoryDouble()
    prefix = "其他规则原样保留。\n" * 1000
    suffix = "\n  例外审批和原有空格均保持。\n"
    original = prefix + "住宿限额为 600 元。" + suffix
    repo.revision["blocks"][0]["text"] = original
    repo.evidence["text"] = "住宿限额为 600 元。"
    _, paused = start_update(repo, update_model())
    proposal = paused["__interrupt__"][0].value["proposal"]
    assert len(original) > 8000
    assert proposal["before"] == original
    assert proposal["after"] == prefix + "住宿限额为 800 元。" + suffix
    assert proposal["after"].encode("utf-8").startswith(prefix.encode("utf-8"))
    assert proposal["after"].endswith(suffix)


@pytest.mark.parametrize("before,after,error", [
    ("500", "800", "does not match"),
    ("600", "600", "unchanged"),
    ("国内出差住宿每晚限额为 600 元。", "", "entire clause"),
])
def test_invalid_model_span_never_creates_a_proposal(before, after, error):
    repo = RepositoryDouble()
    model = update_model(draft_clause=[{"before_span": before, "after_span": after}])
    with pytest.raises(ValueError, match=error):
        start_update(repo, model)
    assert repo.proposals == {}
    assert repo.publish_calls == 0


def test_repeated_amount_needs_unique_original_context():
    original = "国内住宿500元；餐饮500元。"
    with pytest.raises(ValueError, match="uniquely"):
        _apply_span_patch(original, "500", "800")
    assert _apply_span_patch(original, "住宿500", "住宿800") == "国内住宿800元；餐饮500元。"


def test_overlapping_original_matches_are_ambiguous():
    with pytest.raises(ValueError, match="uniquely"):
        _apply_span_patch("aaa", "aa", "b")


def test_patch_can_remove_one_part_without_regenerating_other_text():
    assert _apply_span_patch("原有规则。待删除说明。\n保留规则。", "待删除说明。", "") == "原有规则。\n保留规则。"


def test_full_after_field_is_rejected_by_the_provider_schema():
    model = ScriptedModel(draft_clause=[{"after": "Uncontrolled replacement of the entire policy"}])
    with pytest.raises(ValidationError):
        complete_structured(model, "draft_clause", {})
