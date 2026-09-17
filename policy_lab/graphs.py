"""Bounded LangGraph question answering and reviewed single-clause changes.

Checkpointers are injected by the API lifecycle. Document objects and credentials
never enter graph state. Authorization is refreshed at every node, including both
interrupt nodes when they restart after a resume.
"""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from .llm import ModelTransientError, StructuredLLM, complete_structured


class GraphState(TypedDict, total=False):
    actor_id: str
    question: str
    instruction: str
    query: str
    snapshot: dict
    evidence: list[dict]
    attempts: int
    sufficient: bool
    rewrite: str
    generated: dict
    trace: list[str]
    candidates: list[dict]
    selected: dict | None
    clarification_question: str
    clarifications: list[str]
    clarification_rounds: int
    draft: dict
    proposal_id: str
    approved: bool
    result: dict


def _trace(state, event):
    return state.get("trace", []) + [event]


def _nonempty(value, label, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must contain 1 to {maximum} characters")
    return value.strip()


def _thread_id(config):
    value = config.get("configurable", {}).get("thread_id")
    if not isinstance(value, str) or not value:
        raise PermissionError("An owned workflow thread is required")
    return value


def _actor(repository, state, config, edit=False):
    actor = repository.get_user(state["actor_id"])
    if not actor or not actor.get("active", False):
        raise PermissionError("An active account is required")
    repository.authorize_thread(actor, _thread_id(config))
    if edit and actor.get("role") not in {"manager", "admin"}:
        raise PermissionError("Policy manager permission is required")
    return actor


def _canonical(repository, actor, evidence, current=False):
    """Reload authoritative content; a model can choose an ID but cannot supply text."""
    policy = repository.get_policy(actor, evidence["policy_id"])
    if current and policy["published_revision_id"] != evidence["revision_id"]:
        from .repository import ConflictError

        raise ConflictError("The policy changed; create a new change request")
    if callable(getattr(repository, "get_block", None)):
        # Production reads one normalized block rather than materializing a
        # potentially multi-megabyte revision for each retrieved excerpt.
        block = repository.get_block(actor, evidence["revision_id"], evidence["block_id"])
        revision = {"id": block["revision_id"], "number": block["revision_number"], "policy_id": block["policy_id"]}
    else:
        revision = repository.get_revision(actor, evidence["revision_id"])
        block = next((b for b in revision["blocks"] if b["id"] == evidence["block_id"]), None)
    if revision["policy_id"] != policy["id"] or revision["id"] != evidence["revision_id"]:
        raise ValueError("Evidence revision belongs to a different policy")
    if block is None:
        raise ValueError("The selected canonical clause no longer exists")
    return {
        "id": evidence["id"],
        "policy_id": policy["id"],
        "revision_id": revision["id"],
        "revision_number": revision["number"],
        "block_id": block["id"],
        "title": policy["title"],
        "scope": policy["scope"],
        "heading": block.get("heading", ""),
        "text": block["text"],
        "locator": block.get("locator", {}),
        "score": evidence.get("score", 0),
    }


def _context(evidence, max_total=30000, per_source=4000):
    """Keep exact original excerpts, bounded independently of uploaded file size."""
    bounded, seen, remaining = [], set(), max_total
    for item in evidence:
        if item["id"] in seen or remaining <= 0:
            continue
        copied = dict(item)
        copied["text"] = str(copied["text"])[:min(per_source, remaining)]
        remaining -= len(copied["text"])
        bounded.append(copied)
        seen.add(item["id"])
        if len(bounded) == 12:
            break
    return bounded


def _candidate_cards(candidates):
    return [
        {
            "id": e["id"],
            "policy_id": e["policy_id"],
            "revision_id": e["revision_id"],
            "title": e["title"],
            "heading": e.get("heading", ""),
            "scope": e["scope"],
            "preview": e["text"][:500],
            "summary": f"{e['title']} · {e.get('heading') or e['scope']}",
        }
        for e in candidates
    ]


def _validated_interrupt(payload, validate):
    """A bad reply gets a new interrupt instead of poisoning a saved resume value.

    The API validates before resuming as well. The finite loop is defense in
    depth for other callers; it does not add business-clarification rounds.
    """
    for _ in range(3):
        reply = interrupt(payload)
        error = validate(reply)
        if error is None:
            return reply
        payload = dict(payload, error=error)
    raise ValueError("Too many invalid workflow replies; start a new request")


def _apply_span_patch(original, before_span, after_span):
    """Replace exactly one literal span, including overlapping-match detection."""
    if not isinstance(before_span, str) or not before_span.strip():
        raise ValueError("The proposed original span must be nonempty")
    if not isinstance(after_span, str):
        raise ValueError("The proposed replacement span must be text")
    position = original.find(before_span)
    if position < 0:
        raise ValueError("The proposed span does not match the original clause")
    if original.find(before_span, position + 1) >= 0:
        raise ValueError("The proposed span must uniquely identify one part of the clause")
    # No strip(), normalization, regular expressions or model-supplied offsets:
    # every character outside the selected span is retained verbatim.
    replacement = original[:position] + after_span + original[position + len(before_span):]
    if not replacement.strip():
        raise ValueError("A policy change cannot delete the entire clause")
    if replacement == original:
        raise ValueError("The proposed clause is unchanged")
    return replacement


def build_graphs(repository, checkpointer, llm=None):
    model = llm if llm is not None else StructuredLLM(repository.settings)
    retry = RetryPolicy(max_attempts=2, initial_interval=0.25, retry_on=ModelTransientError)

    def prepare_answer(state, config):
        actor = _actor(repository, state, config)
        question = _nonempty(state["question"], "Question")
        return {"query": question, "snapshot": repository.snapshot(actor), "attempts": 0,
                "trace": ["authorized", "revision_snapshot_pinned"]}

    def retrieve(state, config):
        actor = _actor(repository, state, config)
        evidence = repository.search(actor, state["query"], limit=12, snapshot=state["snapshot"])
        verified = []
        for hit in evidence:
            canonical = _canonical(repository, actor, hit)
            if not hit["text"] or hit["text"] not in canonical["text"]:
                raise ValueError("Search excerpt does not match its canonical source")
            # A matching excerpt may be near the end of a large canonical block.
            # Keep the retrieved span rather than replacing it with the block prefix.
            verified.append(dict(canonical, text=hit["text"]))
        for item in verified:
            if state["snapshot"].get(item["policy_id"]) != item["revision_id"]:
                raise ValueError("Search returned evidence outside the pinned revision snapshot")
        return {"evidence": _context(verified), "attempts": state["attempts"] + 1,
                "trace": _trace(state, "hybrid_retrieval")}

    def assess(state, config):
        _actor(repository, state, config)
        if not state["evidence"]:
            return {"sufficient": False, "rewrite": "", "trace": _trace(state, "no_authorized_evidence")}
        grade = complete_structured(model, "assess_evidence", {
            "question": state["question"], "query": state["query"], "evidence": state["evidence"],
        })
        by_id = {e["id"]: e for e in state["evidence"]}
        if any(i not in by_id for i in grade["relevant_ids"]):
            raise ValueError("The model selected an unknown evidence ID")
        chosen = [by_id[i] for i in dict.fromkeys(grade["relevant_ids"])]
        return {"sufficient": grade["sufficient"] and bool(chosen), "evidence": chosen,
                "rewrite": grade["rewritten_query"].strip(), "trace": _trace(state, "model_evidence_assessment")}

    def after_assess(state):
        if state["sufficient"]:
            return "generate"
        if state["attempts"] < 2 and state.get("rewrite"):
            return "rewrite"
        return "abstain"

    def rewrite(state, config):
        _actor(repository, state, config)
        return {"query": state["rewrite"], "trace": _trace(state, "one_query_rewrite")}

    def generate(state, config):
        actor = _actor(repository, state, config)
        # Recheck access immediately before sending any evidence to the provider.
        for e in state["evidence"]:
            _canonical(repository, actor, e)
        sources = [dict(e, citation_label=f"S{i}") for i, e in enumerate(state["evidence"], 1)]
        generated = complete_structured(model, "answer", {"question": state["question"], "sources": sources})
        return {"generated": generated, "trace": _trace(state, "grounded_answer_draft")}

    def verify(state, config):
        actor = _actor(repository, state, config)
        generated = state["generated"]
        by_label = {f"S{i}": e for i, e in enumerate(state["evidence"], 1)}
        selected = list(dict.fromkeys(generated["citation_ids"]))
        markers = set(re.findall(r"\[(S\d+)\]", generated["answer"]))
        valid = bool(markers) and markers == set(selected) and set(selected) <= by_label.keys()
        if not valid:
            return {"result": {"answer": "无法确认回答中的来源引用，请补充问题后重试。 / Source references could not be verified.",
                               "answerable": False, "citations": [], "trace": _trace(state, "invalid_citations_rejected"),
                               "verification": "citation_ids_and_source_metadata_only"}}
        citations = []
        for label in selected:
            evidence = by_label[label]
            canonical = _canonical(repository, actor, evidence)
            if evidence["text"] not in canonical["text"]:
                raise ValueError("Evidence text does not match its canonical source")
            citations.append(dict(canonical, text=evidence["text"][:1500], citation_label=label))
        return {"result": {"answer": generated["answer"], "answerable": True, "citations": citations,
                           "trace": _trace(state, "citation_metadata_verified"),
                           "verification": "citation_ids_and_source_metadata_only",
                           "version_semantics": "pinned_at_question_start"}}

    def abstain(state, config):
        _actor(repository, state, config)
        return {"result": {"answer": "当前可访问的政策证据不足以确定答案，请补充适用地区、人员或具体场景。 / The accessible policies do not provide enough evidence; clarify the relevant region, role or situation.",
                           "answerable": False, "citations": [], "trace": _trace(state, "insufficient_evidence"),
                           "verification": "citation_ids_and_source_metadata_only"}}

    answer = StateGraph(GraphState)
    for name, fn in [("prepare", prepare_answer), ("retrieve", retrieve), ("assess", assess),
                     ("rewrite", rewrite), ("generate", generate), ("verify", verify), ("abstain", abstain)]:
        answer.add_node(name, fn, retry_policy=retry if name in {"assess", "generate"} else None)
    answer.add_edge(START, "prepare")
    answer.add_edge("prepare", "retrieve")
    answer.add_edge("retrieve", "assess")
    answer.add_conditional_edges("assess", after_assess)
    answer.add_edge("rewrite", "retrieve")
    answer.add_edge("generate", "verify")
    answer.add_edge("verify", END)
    answer.add_edge("abstain", END)

    def prepare_update(state, config):
        _actor(repository, state, config, edit=True)
        instruction = _nonempty(state["instruction"], "Change instruction")
        return {"query": instruction, "clarifications": [], "clarification_rounds": 0,
                "selected": None, "trace": ["manager_authorized"]}

    def locate(state, config):
        actor = _actor(repository, state, config, edit=True)
        candidates = _context(repository.search(actor, state["query"], limit=8, editable=True))
        selection = complete_structured(model, "select_clause", {
            "instruction": state["instruction"], "clarifications": state["clarifications"],
            "candidates": candidates,
        }) if candidates else {"selected_id": None, "question": "请描述要调整的具体规定及适用人员或地区。 / Describe the rule and who or where it applies to."}
        selected = next((e for e in candidates if e["id"] == selection["selected_id"]), None)
        if selection["selected_id"] and selected is None:
            raise ValueError("The model selected an unknown policy clause")
        if selection["question"]:
            selected = None
        return {"candidates": candidates, "selected": selected,
                "clarification_question": selection["question"] or "请确认要调整的业务范围。 / Clarify the intended business scope.",
                "trace": _trace(state, "editable_clause_candidates_located")}

    def target_route(state):
        if state.get("selected"):
            return "draft"
        return "clarify" if state["clarification_rounds"] < 2 else "needs_clarification"

    def clarify(state, config):
        actor = _actor(repository, state, config, edit=True)
        for e in state.get("candidates", []):
            _canonical(repository, actor, e, current=True)
        def validate_reply(reply):
            if not isinstance(reply, dict) or bool(reply.get("choice_id")) == bool(reply.get("clarification")):
                return "Provide either a candidate choice or a business clarification"
            if reply.get("choice_id"):
                if not any(e["id"] == reply["choice_id"] for e in state["candidates"]):
                    return "Select one of the displayed candidates"
            elif not isinstance(reply["clarification"], str) or not 1 <= len(reply["clarification"].strip()) <= 2000:
                return "Clarification must contain 1 to 2000 characters"
            return None

        reply = _validated_interrupt(
            {"type": "clarification", "question": state["clarification_question"],
             "candidates": _candidate_cards(state.get("candidates", []))}, validate_reply,
        )
        # On resume this node starts again, including the authorization above.
        if not isinstance(reply, dict) or bool(reply.get("choice_id")) == bool(reply.get("clarification")):
            raise ValueError("Provide either a candidate choice or a business clarification")
        rounds = state["clarification_rounds"] + 1
        if reply.get("choice_id"):
            chosen = next((e for e in state["candidates"] if e["id"] == reply["choice_id"]), None)
            if chosen is None:
                raise ValueError("Select one of the displayed candidates")
            chosen = _canonical(repository, actor, chosen, current=True)
            return {"selected": chosen, "clarification_rounds": rounds,
                    "trace": _trace(state, "candidate_chosen_by_manager")}
        clarification = _nonempty(reply["clarification"], "Clarification", 2000)
        clarifications = state["clarifications"] + [clarification]
        return {"selected": None, "clarifications": clarifications, "clarification_rounds": rounds,
                "query": state["instruction"] + "\n" + "\n".join(clarifications),
                "trace": _trace(state, "business_scope_clarified")}

    def draft(state, config):
        actor = _actor(repository, state, config, edit=True)
        selected = _canonical(repository, actor, state["selected"], current=True)
        if len(selected["text"]) > 20000:
            raise ValueError("This clause exceeds the safe edit size; ingest a document with smaller structured sections")
        proposed = complete_structured(model, "draft_clause", {
            "instruction": state["instruction"], "clarifications": state["clarifications"], "clause": selected,
        })
        if proposed["question"] or proposed["before_span"] is None or proposed["after_span"] is None:
            return {"selected": None, "draft": {},
                    "clarification_question": proposed["question"] or "请说明该条规定应如何变化。 / Describe the intended change.",
                    "trace": _trace(state, "draft_requires_clarification")}
        replacement = _apply_span_patch(selected["text"], proposed["before_span"], proposed["after_span"])
        return {"selected": selected, "draft": {"after": replacement, "reason": proposed["reason"]},
                "trace": _trace(state, "unique_literal_span_patched")}

    def draft_route(state):
        if state.get("draft"):
            return "propose"
        return "clarify" if state["clarification_rounds"] < 2 else "needs_clarification"

    def propose(state, config):
        actor = _actor(repository, state, config, edit=True)
        evidence = _canonical(repository, actor, state["selected"], current=True)
        proposal = repository.create_proposal(
            actor, state["instruction"], evidence, state["draft"]["after"], state["draft"]["reason"],
            request_key=_thread_id(config),
        )
        return {"proposal_id": proposal["id"], "trace": _trace(state, "immutable_proposal_created")}

    def review(state, config):
        actor = _actor(repository, state, config, edit=True)
        proposal = repository.get_proposal(actor, state["proposal_id"])
        source = _canonical(repository, actor, state["selected"])
        proposal = dict(proposal, policy_title=source["title"], heading=source["heading"],
                        base_revision_number=source["revision_number"])
        def validate_decision(decision):
            if not isinstance(decision, dict) or type(decision.get("approve")) is not bool:
                return "Review requires an explicit approve true or false"
            return None

        decision = _validated_interrupt({"type": "review", "proposal": proposal}, validate_decision)
        return {"approved": decision["approve"], "trace": _trace(state, "manager_review_received")}

    def publish(state, config):
        actor = _actor(repository, state, config, edit=True)
        job = repository.publish_proposal(actor, state["proposal_id"])
        return {"result": {"status": "queued", "proposal_id": state["proposal_id"], "job": job,
                           "trace": _trace(state, "publication_job_queued")}}

    def reject(state, config):
        actor = _actor(repository, state, config, edit=True)
        repository.reject_proposal(actor, state["proposal_id"])
        return {"result": {"status": "rejected", "proposal_id": state["proposal_id"],
                           "trace": _trace(state, "proposal_rejected")}}

    def needs_clarification(state, config):
        _actor(repository, state, config, edit=True)
        return {"result": {"status": "needs_clarification", "question": state["clarification_question"],
                           "message": "请提交更明确的修改要求。 / Start a new request with a more specific change.",
                           "trace": _trace(state, "clarification_budget_exhausted")}}

    update = StateGraph(GraphState)
    for name, fn in [("prepare", prepare_update), ("locate", locate), ("clarify", clarify), ("draft", draft),
                     ("propose", propose), ("review", review), ("publish", publish), ("reject", reject),
                     ("needs_clarification", needs_clarification)]:
        update.add_node(name, fn, retry_policy=retry if name in {"locate", "draft"} else None)
    update.add_edge(START, "prepare")
    update.add_edge("prepare", "locate")
    update.add_conditional_edges("locate", target_route)
    update.add_conditional_edges("clarify", lambda s: "draft" if s.get("selected") else "locate")
    update.add_conditional_edges("draft", draft_route)
    update.add_edge("propose", "review")
    update.add_conditional_edges("review", lambda s: "publish" if s["approved"] else "reject")
    for node in ["publish", "reject", "needs_clarification"]:
        update.add_edge(node, END)
    return answer.compile(checkpointer=checkpointer), update.compile(checkpointer=checkpointer)
