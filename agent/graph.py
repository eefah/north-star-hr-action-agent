"""Stage 3: the LangGraph workflow that runs the Stage 2 tools for one employee request.

Two kinds of request:
    leave_request    eligibility -> dates -> draft -> employee approval -> save the approved draft locally
    policy_question  a grounded, cited answer (or a refusal)

Design decisions
- Deterministic workflow. No LLM chooses the next step: routing reads tool result codes. (An LLM planner
  can sit in front of this later; it would only fill the request.)
- Saved state is plain JSON data only: the request, result codes, evidence references, decisions,
  approval status, the draft and the trace. Live clients (ToolContext, retriever, LLM) and the drafts
  folder are passed per run as LangGraph run-time context, which is never checkpointed. Every node's
  update is checked with json.dumps before it is saved.
- Interrupts pause the workflow for the employee: missing details, a retry offer after a temporary
  fault, and approval of the draft. A node restarts from its beginning when resumed, so no node does
  anything consequential before an interrupt.
- The only consequential action is save_draft: writing the approved draft to a local folder. It runs
  only after the employee approves that exact draft ID, and it is idempotent (exclusive file creation),
  so resuming or replaying can't save twice. Nothing is sent to PeopleHub, a manager or anyone else.
- Faults: tools already return TOOL_FAILURE instead of raising, and retry transient errors (bounded).
  The graph adds one bounded recovery loop: after a temporary fault the employee may retry the step,
  at most MAX_RETRY_ROUNDS times. Every node also has a safety net: an unexpected exception becomes a
  cannot_determine outcome with an approved message, and the details go to the restricted diagnostic log.
- Three outcomes (agent/explain.py): can_proceed, cannot_proceed (a known rule or invalid dates, with the
  reason) and cannot_determine (missing evidence or a fault; never described as ineligibility).
- If the approval arrives on a later day than the draft was prepared, the draft is prepared again
  under today's date (the notice period may have run out) and must be approved again.
"""
import json
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from agent import diagnostics
from agent.explain import CAN_PROCEED, CANNOT_DETERMINE, FIELD_NAMES, explain
from agent.tools import ERROR_MESSAGES, Code, ToolContext, run_tool
from agent.rules import PEOPLE_TEAM

PROFILE_FIELDS = ("employment_type", "uk_based", "working_pattern")
LEAVE_FIELDS = PROFILE_FIELDS + ("start_date", "end_date", "employee_note")
REQUEST_FIELDS = {"leave_request": set(LEAVE_FIELDS), "policy_question": {"question"}}

# Statuses. The first group are pauses waiting for the employee; the second group are terminal.
AWAITING_INFO, AWAITING_RETRY, AWAITING_APPROVAL = "awaiting_employee_info", "awaiting_retry", "awaiting_approval"
COMPLETED, ANSWERED, REFUSED, UNDETERMINED, CANCELLED = "completed", "answered", "refused", "undetermined", \
    "cancelled"
TERMINAL_STATUSES = {COMPLETED, ANSWERED, REFUSED, UNDETERMINED, CANCELLED}
SAVED_STATUS = "approved_by_employee_not_submitted"


@dataclass(frozen=True)
class GraphContext:
    """Per-run, never saved. Built by application code."""
    tools: ToolContext
    drafts_dir: Path
    max_info_rounds: int = 2
    max_retry_rounds: int = 2
    max_invalid_replies: int = 3


def _merge_refs(old: list, new: list) -> list:
    merged = {r["chunk_id"]: r for r in old}
    for r in new:
        merged.setdefault(r["chunk_id"], r)
    return list(merged.values())


class State(TypedDict, total=False):
    kind: str
    request: dict
    request_date: str
    status: str
    route: str  # next node, set by the node that just ran
    results: dict  # step -> ToolResult.to_dict() of the latest run of that step
    decisions: dict  # step -> result code
    evidence_refs: Annotated[list, _merge_refs]  # verified policy chunks the decisions relied on
    trace: Annotated[list, operator.add]
    draft: dict | None
    approval: dict
    retry_rounds: int
    retry_target: str | None
    info_rounds: int
    outcome: dict | None
    actions: Annotated[list, operator.add]  # consequential actions performed, for audit and duplicate checks


# ---------------------------------------------------------------- helpers

def _tool_step(rt: Runtime[GraphContext], state: State, step: str, tool: str, args: dict) -> tuple[Any, dict]:
    """Run one tool and record its result, trace entry, decision and evidence references."""
    ctx = rt.context.tools
    r = run_tool(ctx, tool, args)
    d = r.to_dict()
    refs = [{"chunk_id": cid, "doc_id": c.get("parent_doc_id"), "title": c.get("title"),
             "version": c.get("version"), "status": c.get("status"), "effective_from": c.get("effective_from")}
            for cid in r.citations if (c := ctx.evidence.chunks.get(cid))]
    entry = {"step": step, "tool": tool, "code": r.code.value, "reason": r.reason.value, "citations": r.citations,
             "assumptions": r.assumptions, "attempts": r.attempts, "errors": r.errors,
             "failure": d["data"].get("failure"), "reference": d["data"].get("reference")}
    update = {"results": {**state.get("results", {}), step: d},
              "decisions": {**state.get("decisions", {}), step: r.reason.value},
              "evidence_refs": refs, "trace": [entry]}
    return r, update


def _conclude(result: dict, status: str | None = None) -> dict:
    outcome = explain(result)
    if status is None:
        status = {CAN_PROCEED: COMPLETED, CANNOT_DETERMINE: UNDETERMINED}.get(outcome["category"], REFUSED)
    return {"outcome": outcome, "status": status, "route": "finish"}


def _after_failure(rt: Runtime[GraphContext], state: State, step: str, result: dict) -> dict:
    """Offer a retry for a temporary fault while rounds remain; otherwise conclude."""
    if result["data"].get("retryable") and state.get("retry_rounds", 0) < rt.context.max_retry_rounds:
        return {"route": "offer_retry", "retry_target": step, "status": AWAITING_RETRY}
    return _conclude(result)


def _event(step: str, **fields) -> dict:
    return {"step": step, **fields}


def _safe(name: str):
    """Safety net for every node: an unexpected exception becomes a controlled cannot_determine outcome.
    Interrupts (GraphBubbleUp) pass through: they are how the graph pauses, not errors."""
    def wrap(fn):
        def node(state: State, runtime: Runtime[GraphContext]):
            try:
                update = fn(state, runtime)
                json.dumps(update)  # only plain data may be saved in a checkpoint
                return update
            except GraphBubbleUp:
                raise
            except Exception as e:
                # Self-contained on purpose: no explain() or other helper that could itself be failing.
                ref = diagnostics.new_ref()
                diagnostics.record(ref, "internal", e, tool=f"graph:{name}")
                outcome = {"category": CANNOT_DETERMINE, "code": Code.TOOL_FAILURE.value, "citations": [],
                           "message": ERROR_MESSAGES["unavailable"].format(people_team=PEOPLE_TEAM, ref=ref),
                           "recovery": f"If it keeps happening, contact HR: the People team at {PEOPLE_TEAM}, "
                                       f"quoting reference {ref}.", "retryable": False}
                return {"outcome": outcome, "status": UNDETERMINED, "route": "finish",
                        "trace": [{"step": name, "code": Code.TOOL_FAILURE.value, "failure": "internal",
                                   "errors": [f"internal {type(e).__name__} (diagnostic ref {ref})"]}]}
        node.__name__ = name
        return node
    return wrap


# ---------------------------------------------------------------- nodes

@_safe("intake")
def intake(state: State, rt: Runtime[GraphContext]) -> dict:
    kind, request = state.get("kind"), state.get("request")
    base = {"request_date": rt.context.tools.today.isoformat(), "results": {}, "decisions": {}, "draft": None,
            "approval": {"status": "not_requested"}, "retry_rounds": 0, "retry_target": None, "info_rounds": 0,
            "outcome": None, "trace": [_event("intake", kind=kind)]}
    problems = []
    if kind not in REQUEST_FIELDS:
        problems.append(f"unknown request kind {kind!r}")
    elif not isinstance(request, dict):
        problems.append("request must be an object")
    else:
        problems += [f"unexpected field {k!r}" for k in request if k not in REQUEST_FIELDS[kind]]
    if problems:  # e.g. a 'roles' field: the request can never set roles or evidence
        bad = {"tool": "intake", "code": Code.INVALID_ARGUMENTS.value, "ok": False, "citations": [],
               "message": "; ".join(problems), "data": {"problems": problems}}
        return {**base, **_conclude(bad), "trace": base["trace"] + [_event("intake", code=bad["code"],
                                                                             problems=problems)]}
    return {**base, "status": "running", "route": "answer" if kind == "policy_question" else "eligibility"}


@_safe("answer")
def answer(state: State, rt: Runtime[GraphContext]) -> dict:
    r, update = _tool_step(rt, state, "answer", "answer_policy", {"question": state["request"].get("question")})
    if r.code == Code.OK:
        return {**update, **_conclude(r.to_dict(), ANSWERED)}
    if r.code == Code.TOOL_FAILURE:
        return {**update, **_after_failure(rt, state, "answer", r.to_dict())}
    return {**update, **_conclude(r.to_dict())}


@_safe("eligibility")
def eligibility(state: State, rt: Runtime[GraphContext]) -> dict:
    req = state["request"]
    r, update = _tool_step(rt, state, "eligibility", "check_eligibility",
                           {k: req.get(k) for k in PROFILE_FIELDS})
    if r.code == Code.OK:
        return {**update, "route": "dates"}
    if r.code == Code.MISSING_EMPLOYEE_INFO and state.get("info_rounds", 0) < rt.context.max_info_rounds:
        return {**update, "route": "ask_info", "status": AWAITING_INFO}
    if r.code == Code.TOOL_FAILURE:
        return {**update, **_after_failure(rt, state, "eligibility", r.to_dict())}
    return {**update, **_conclude(r.to_dict())}


@_safe("ask_info")
def ask_info(state: State, rt: Runtime[GraphContext]) -> dict:
    missing = state["results"]["eligibility"]["data"]["missing"]
    prompt = {"kind": "employee_info", "missing": missing,
              "question": "Please tell me " + "; ".join(FIELD_NAMES.get(f, f) for f in missing) + "."}
    for _ in range(rt.context.max_invalid_replies):
        reply = interrupt(prompt)
        if isinstance(reply, dict) and reply and set(reply) <= set(missing):
            return {"request": {**state["request"], **reply}, "info_rounds": state.get("info_rounds", 0) + 1,
                    "status": "running", "route": "eligibility",
                    "trace": [_event("ask_info", supplied=sorted(reply))]}
        prompt = {**prompt, "error": f"Only these details can be given here: {', '.join(missing)}."}
    bad = {"tool": "ask_info", "code": Code.INVALID_ARGUMENTS.value, "ok": False, "citations": [],
           "message": "no usable reply", "data": {}}
    return {**_conclude(bad), "trace": [_event("ask_info", code=bad["code"])]}


@_safe("dates")
def dates(state: State, rt: Runtime[GraphContext]) -> dict:
    req = state["request"]
    r, update = _tool_step(rt, state, "dates", "calculate_dates",
                           {"start_date": req.get("start_date"), "end_date": req.get("end_date")})
    if r.code == Code.OK:
        return {**update, "route": "draft"}
    if r.code == Code.TOOL_FAILURE:
        return {**update, **_after_failure(rt, state, "dates", r.to_dict())}
    return {**update, **_conclude(r.to_dict())}


@_safe("draft")
def draft(state: State, rt: Runtime[GraphContext]) -> dict:
    """Prepare the draft in memory (prepare_draft re-runs both checks itself). Nothing is saved here."""
    req = state["request"]
    args = {k: req.get(k) for k in PROFILE_FIELDS + ("start_date", "end_date")}
    if req.get("employee_note"):
        args["employee_note"] = req["employee_note"]
    r, update = _tool_step(rt, state, "draft", "prepare_draft", args)
    if r.code == Code.OK:
        d = r.data["draft"]
        outcome = explain(r.to_dict()) | {
            "message": "You can go ahead: you are eligible, the dates meet the notice rule, and a draft is ready "
                       "for you to review. Nothing has been saved or submitted.",
            "recovery": None}
        return {**update, "draft": d, "outcome": outcome, "status": AWAITING_APPROVAL, "route": "approval",
                "request_date": rt.context.tools.today.isoformat(),
                "approval": {"status": "pending", "draft_id": d["draft_id"]}}
    if r.code == Code.TOOL_FAILURE:
        return {**update, **_after_failure(rt, state, "draft", r.to_dict())}
    return {**update, **_conclude(r.to_dict())}


@_safe("approval")
def approval(state: State, rt: Runtime[GraphContext]) -> dict:
    """Human approval. Only the signed-in employee can approve, and only the exact draft shown."""
    d = state["draft"]
    prompt = {"kind": "approval", "draft_id": d["draft_id"], "draft": d,
              "question": "Approve this draft so it is saved for you to submit in PeopleHub yourself, or reject it."}
    employee = rt.context.tools.employee_ref
    for _ in range(rt.context.max_invalid_replies):
        reply = interrupt(prompt)
        ok = isinstance(reply, dict) and reply.get("decision") in ("approve", "reject")
        if ok and reply.get("draft_id") == d["draft_id"] and reply.get("by") == employee:
            if reply["decision"] == "reject":
                cancelled = {"status": "rejected", "draft_id": d["draft_id"], "by": employee}
                return {"approval": cancelled, "status": CANCELLED, "route": "finish", "draft": d,
                        "outcome": {"category": "cancelled", "code": "CANCELLED", "citations": [],
                                    "message": "You rejected the draft, so nothing was saved or submitted.",
                                    "recovery": "Start a new request if you change your mind.", "retryable": False},
                        "trace": [_event("approval", decision="reject", draft_id=d["draft_id"])]}
            today = rt.context.tools.today.isoformat()
            if today != state["request_date"]:  # a later day: the notice rule must be checked again
                return {"approval": {"status": "expired", "draft_id": d["draft_id"]}, "route": "draft",
                        "status": "running", "trace": [_event("approval", decision="recheck_needed",
                                                              prepared_on=state["request_date"], today=today)]}
            approved = {"status": "approved", "draft_id": d["draft_id"], "by": employee, "on": today}
            return {"approval": approved, "route": "save_draft", "status": "running",
                    "trace": [_event("approval", decision="approve", draft_id=d["draft_id"])]}
        why = "This approval doesn't match the draft shown or the signed-in employee." if ok else \
            "Reply with decision 'approve' or 'reject', the draft_id and who you are."
        prompt = {**prompt, "error": why}
    bad = {"tool": "approval", "code": Code.INVALID_ARGUMENTS.value, "ok": False, "citations": [],
           "message": "no valid approval", "data": {}}
    return {**_conclude(bad), "approval": {"status": "not_given", "draft_id": d["draft_id"]},
            "trace": [_event("approval", code=bad["code"])]}


@_safe("save_draft")
def save_draft(state: State, rt: Runtime[GraphContext]) -> dict:
    """The only consequential action: save the approved draft locally. Idempotent by draft ID."""
    d, a = state["draft"], state["approval"]
    if a.get("status") != "approved" or a.get("draft_id") != d["draft_id"]:  # defence in depth
        raise RuntimeError("save_draft reached without approval of this draft")
    record = {**d, "status": SAVED_STATUS, "approved_by": a["by"], "approved_on": a["on"]}
    body = json.dumps(record, indent=2, sort_keys=True)
    folder = rt.context.drafts_dir
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = folder / f"{d['draft_id']}.json"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        action = {"action": "save_draft", "draft_id": d["draft_id"], "result": "saved"}
    except FileExistsError:
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError("a different draft is already saved under this ID")
        action = {"action": "save_draft", "draft_id": d["draft_id"], "result": "already_saved"}
    next_steps = " ".join(d.get("next_steps", []))
    outcome = {"category": CAN_PROCEED, "code": Code.OK.value, "citations": d.get("citations", []),
               "message": "Your approved draft is saved for your records. It has NOT been submitted to PeopleHub "
                          f"and is NOT approved by your line manager. Next steps from the policy: {next_steps}",
               "recovery": None, "retryable": False}
    return {"status": COMPLETED, "route": "finish", "outcome": outcome, "actions": [action],
            "trace": [_event("save_draft", **action)]}


@_safe("offer_retry")
def offer_retry(state: State, rt: Runtime[GraphContext]) -> dict:
    step = state["retry_target"]
    failed = state["results"][step]
    reply = interrupt({"kind": "retry", "message": failed["message"],
                       "question": "Try again now? Reply {'retry': true} or {'retry': false}."})
    if isinstance(reply, dict) and reply.get("retry") is True:
        return {"retry_rounds": state.get("retry_rounds", 0) + 1, "status": "running", "route": step,
                "trace": [_event("offer_retry", retry=True, target=step)]}
    return {**_conclude(failed), "trace": [_event("offer_retry", retry=False, target=step)]}


@_safe("finish")
def finish(state: State, rt: Runtime[GraphContext]) -> dict:
    if state.get("outcome") is None or state.get("status") not in TERMINAL_STATUSES:
        raise RuntimeError(f"finished without a terminal outcome (status {state.get('status')!r})")
    return {"trace": [_event("finish", status=state["status"], category=state["outcome"]["category"],
                             code=state["outcome"]["code"])]}


# ---------------------------------------------------------------- graph

# Explicit transitions: each node may only go to the nodes listed here.
TRANSITIONS = {
    "intake": ["answer", "eligibility", "finish"],
    "answer": ["offer_retry", "finish"],
    "eligibility": ["dates", "ask_info", "offer_retry", "finish"],
    "ask_info": ["eligibility", "finish"],
    "dates": ["draft", "offer_retry", "finish"],
    "draft": ["approval", "offer_retry", "finish"],
    "approval": ["save_draft", "draft", "finish"],
    "save_draft": ["finish"],
    "offer_retry": ["answer", "eligibility", "dates", "draft", "finish"],
}
NODES = {"intake": intake, "answer": answer, "eligibility": eligibility, "ask_info": ask_info, "dates": dates,
         "draft": draft, "approval": approval, "save_draft": save_draft, "offer_retry": offer_retry,
         "finish": finish}


def build_graph(checkpointer):
    g = StateGraph(State, context_schema=GraphContext)
    for name, fn in NODES.items():
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    for name, targets in TRANSITIONS.items():
        g.add_conditional_edges(name, lambda s: s["route"], {t: t for t in targets})
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- application entry points

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _invoke(app, payload, thread_id: str, ctx: GraphContext) -> dict | None:
    """Run the graph. Nodes already contain their own faults; this catches anything LangGraph itself raises,
    so the application never receives an uncaught exception. The checkpoint is left as it was."""
    try:
        app.invoke(payload, _config(thread_id), context=ctx)
        return None
    except Exception as e:
        ref = diagnostics.new_ref()
        diagnostics.record(ref, "internal", e, tool="graph:invoke")
        return {"message": ERROR_MESSAGES["unavailable"].format(people_team=PEOPLE_TEAM, ref=ref),
                "reason": Code.TOOL_FAILURE.value, "category": CANNOT_DETERMINE}


def start(app, ctx: GraphContext, thread_id: str, kind: str, request: dict) -> dict:
    if app.get_state(_config(thread_id)).values:
        raise ValueError(f"thread {thread_id!r} already exists; resume it instead")
    fault = _invoke(app, {"kind": kind, "request": request}, thread_id, ctx)
    return employee_view(app, thread_id) | (fault or {})


def resume(app, ctx: GraphContext, thread_id: str, reply: Any) -> dict:
    """Continue a paused workflow with the employee's reply. A finished workflow is left as it is."""
    snap = app.get_state(_config(thread_id))
    if not snap.values:
        raise ValueError(f"no workflow {thread_id!r}")
    fault = None
    if snap.next:
        # An empty reply is passed as a marker string: LangGraph reads an empty dict as "no answer" and never
        # delivers it. Every node treats a non-dict reply as invalid.
        no_reply = reply is None or (isinstance(reply, dict) and not reply)
        fault = _invoke(app, Command(resume="(no reply)" if no_reply else reply), thread_id, ctx)
    return employee_view(app, thread_id) | (fault or {})


def employee_view(app, thread_id: str) -> dict:
    """What the employee may see: status, explanation, the pending question and the draft.
    Never the trace, errors, diagnostic details or raw tool data."""
    snap = app.get_state(_config(thread_id))
    s = snap.values
    pending = None
    for task in snap.tasks:
        for i in task.interrupts:
            pending = {k: v for k, v in i.value.items() if k != "draft"}
    outcome = s.get("outcome") or {}
    return {"status": s.get("status"), "category": outcome.get("category"), "reason": outcome.get("code"),
            "message": outcome.get("message"), "recovery": outcome.get("recovery"),
            "citations": outcome.get("citations", []), "pending": pending,
            "draft": s.get("draft") if s.get("status") in (AWAITING_APPROVAL, COMPLETED) else None}
