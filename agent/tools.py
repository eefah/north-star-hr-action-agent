"""Agent tools: the existing RAG pipeline and the Stage 1 rules, wrapped for the agent.

Trust boundary. Everything the LLM or the employee can influence arrives as tool *arguments*, which
run_tool() validates against TOOL_SPECS: unknown arguments are refused, so no argument can set roles,
policy status, version or evidence. Trusted values (the signed-in user's roles, today's date, the
policy date, the retriever and LLM clients) live in ToolContext, which only application code builds.

Evidence. Rules are applied only to chunk records the role-filtered retriever returned for this user
(EvidenceStore), re-checked locally against the same filter (tenant, roles, status, effective dates),
then verified by rules.verify_evidence (policy ID, version, status, verbatim supporting text).

Results. Every tool returns a ToolResult with a machine-readable Code, so the Stage 3 graph can route
on result.code without reading prose.

Failures. Only errors likely to recover (timeouts, connection failures, HTTP 408/425/429/5xx) are
retried. A service that rejects the request (bad credentials, malformed request) stops at once. Anything
else is treated as a code defect: run_tool catches it as a last resort. All three return TOOL_FAILURE,
never a business code like NOT_ELIGIBLE, so a fault is never shown as a decision. Employees see only
ERROR_MESSAGES. Technical details go to the restricted diagnostic log (agent/diagnostics.py), never into
the result, because the trace is not private by default.
"""
import copy
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Callable

import groq
import httpx
from pinecone.errors import PineconeConnectionError, PineconeError

import config
from agent import diagnostics
from agent.dates import DATE_FORMAT_HINT, WORKING_DAY_ASSUMPTION, BankHolidays, parse_date
from agent.rules import APPROVER_ROLES, CONTRACTOR_EXCLUDED, DATE_RULES, ELIGIBILITY_RULES, \
    EMPLOYMENT_TYPE_NOT_COVERED, EVIDENCE_UNVERIFIED, LOCATION_NOT_COVERED, PEOPLE_TEAM, POLICY_DOC_ID, RULES, \
    EmployeeProfile, assess_dates, check_eligibility, verify_evidence
from retriever import HybridRetriever


class Code(str, Enum):
    OK = "OK"
    NO_EVIDENCE = "NO_EVIDENCE"  # retrieval found nothing relevant this user may see
    INVALID_CITATIONS = "INVALID_CITATIONS"  # the answer cites IDs that weren't retrieved
    MISSING_EMPLOYEE_INFO = "MISSING_EMPLOYEE_INFO"  # ask the employee: data["missing"] lists the fields
    NOT_ELIGIBLE = "NOT_ELIGIBLE"  # the policy excludes this employee (contractors)
    UNSUPPORTED_EMPLOYMENT_TYPE = "UNSUPPORTED_EMPLOYMENT_TYPE"  # policy is silent: People team
    UNSUPPORTED_LOCATION = "UNSUPPORTED_LOCATION"  # not UK-based; policy is silent: People team
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # required policy text missing, stale or unauthorised
    NOTICE_TOO_SHORT = "NOTICE_TOO_SHORT"  # data["earliest_valid_start"] is the alternative to offer
    INVALID_DATES = "INVALID_DATES"  # ask the employee for different dates
    PRECONDITION_FAILED = "PRECONDITION_FAILED"  # e.g. drafting before eligibility and dates are OK
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"  # the planner sent bad or forbidden arguments
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    TOOL_FAILURE = "TOOL_FAILURE"  # a fault, not a decision: data["failure"] and data["retryable"] say which


@dataclass
class ToolResult:
    tool: str
    code: Code
    message: str  # plain English, for the trace and the explanation step
    data: dict = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    attempts: int = 0  # calls to external services (Pinecone, Groq), retries included; 0 if none were needed
    # Every failed attempt, as exception type and diagnostic reference only, never exception text.
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.code == Code.OK

    @property
    def reason(self) -> Code:
        """The code to explain to the employee: for a refused draft, the failing check's own code."""
        if self.code == Code.PRECONDITION_FAILED and "reason_code" in self.data:
            return Code(self.data["reason_code"])
        return self.code

    def to_dict(self) -> dict:
        return asdict(self) | {"code": self.code.value, "ok": self.ok}


# Fixed queries used to fetch the policy sections a rule needs. Written by us, never by the LLM.
EVIDENCE_QUERIES = {
    "annual_leave#eligibility": "Who is eligible for annual leave and how many days do permanent UK employees get?",
    "annual_leave#how-to-request": "How do I request annual leave, how much notice is needed and who approves it?",
    "annual_leave#contact": "Who owns the annual leave policy and does it contain individual leave balances?",
}
BALANCE_DISCLAIMER = ("This is the number of working days in the requested dates, under the assumption shown. "
                      "It is not your remaining leave balance, which the agent cannot see.")
DRAFT_STATUS = "draft_pending_employee_review"
DRAFT_STATUS_TEXT = ("Draft prepared by the HR Policy Action Agent. It has NOT been submitted to PeopleHub and has "
                     "NOT been approved by your line manager. Approving this draft only confirms it is ready "
                     "for you to submit in PeopleHub yourself.")
MAX_TEXT = 500
MAX_NOTE = 300

# The only failure wording an employee sees. Draft wording, pending approval.
ERROR_MESSAGES = {
    "transient": ("We couldn't reach the HR policy service just now, so nothing has been decided about your "
                  "request. Please try again in a few minutes."),
    "unavailable": ("The HR policy assistant can't complete this right now, so nothing has been decided about "
                    "your request. This is not a decision about your leave. Please try again later, or contact "
                    "the People team at {people_team} and quote reference {ref}."),
}


@dataclass
class EvidenceStore:
    """Chunk records returned by the role-filtered retriever for this user, by chunk ID."""
    chunks: dict[str, dict] = field(default_factory=dict)

    def add(self, chunk: dict) -> None:
        self.chunks[chunk["_id"]] = copy.deepcopy(chunk)  # a copy, so nothing downstream can alter it


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    backoff_s: tuple[float, ...] = (0.5, 1.5)


@dataclass(frozen=True)
class ToolContext:
    """Trusted, built by application code only, and immutable once built. Never populated from LLM or
    employee text. The evidence store inside it is filled only by retrieval.

    Never put a ToolContext in saved graph state: it holds live clients and callables. Saved state keeps
    only plain data (employee details, evidence references, decisions, result codes); the graph gets
    the context from run-time configuration."""
    roles: tuple[str, ...]
    today: date  # the request date
    retriever: HybridRetriever
    rag: Any  # rag.RAG, or a test double with the same answer() method
    holidays: BankHolidays
    employee_ref: str = "synthetic-employee"
    as_of: date | None = None  # policies in force on this date; defaults to today
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    retry: RetryPolicy = RetryPolicy()
    sleep: Callable[[float], None] = time.sleep
    new_id: Callable[[], str] = lambda: uuid.uuid4().hex[:12]

    def __post_init__(self):
        object.__setattr__(self, "roles", tuple(self.roles))

    @property
    def policy_date(self) -> date:
        return self.as_of or self.today

    @property
    def filter(self) -> dict:
        return HybridRetriever.build_filter(list(self.roles), self.policy_date)


_TRANSIENT_TYPES = (TimeoutError, ConnectionError, PineconeConnectionError, groq.APIConnectionError,
                    httpx.TransportError)  # SDK timeouts subclass TimeoutError or APIConnectionError
_TRANSIENT_STATUS = {408, 425, 429}  # plus every 5xx


def classify_error(e: Exception) -> str:
    """'transient' (worth retrying), 'rejected' (the service refused the request: retrying can't help)
    or 'defect' (not a service error at all: a bug, so it goes to the run_tool safety net)."""
    if isinstance(e, _TRANSIENT_TYPES):
        return "transient"
    status = getattr(e, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return "transient" if status in _TRANSIENT_STATUS or status >= 500 else "rejected"
    if isinstance(e, (PineconeError, groq.GroqError)):  # bad config or request values, unreadable response
        return "rejected"
    return "defect"


class _Failed(Exception):
    def __init__(self, kind: str, attempts: int, errors: list[str], ref: str):
        self.kind, self.attempts, self.errors, self.ref = kind, attempts, errors, ref


def _with_retry(ctx: ToolContext, what: str, fn: Callable[[], Any]) -> tuple[Any, int, list[str]]:
    """Call an external service. Retries transient errors with backoff; stops at once if the service rejects
    the request; lets code defects propagate. Raises _Failed when it gives up."""
    errors, ref = [], diagnostics.new_ref()
    for attempt in range(1, ctx.retry.attempts + 1):
        try:
            return fn(), attempt, errors
        except Exception as e:
            kind = classify_error(e)
            if kind == "defect":
                raise
            diagnostics.record(ref, kind, e, tool=what, attempt=attempt)
            errors.append(f"attempt {attempt}: {kind} {type(e).__name__} (diagnostic ref {ref})")
            if kind == "rejected":
                raise _Failed(kind, attempt, errors, ref) from None
            if attempt < ctx.retry.attempts:
                ctx.sleep(ctx.retry.backoff_s[min(attempt - 1, len(ctx.retry.backoff_s) - 1)])
    raise _Failed("transient", ctx.retry.attempts, errors, ref)


def _failure(tool: str, f: _Failed) -> ToolResult:
    """A fault, reported as TOOL_FAILURE with an approved message. Never a business outcome."""
    retryable = f.kind == "transient"
    message = ERROR_MESSAGES["transient"] if retryable else \
        ERROR_MESSAGES["unavailable"].format(people_team=PEOPLE_TEAM, ref=f.ref)
    return ToolResult(tool, Code.TOOL_FAILURE, message,
                      data={"failure": f.kind, "retryable": retryable, "reference": f.ref},
                      attempts=f.attempts, errors=f.errors)


def _authorised(ctx: ToolContext, chunk: dict) -> bool:
    return HybridRetriever.matches(chunk, ctx.filter)


def _hit_summary(h) -> dict:
    m = h.meta
    return {"id": h.id, "title": m.get("title"), "heading": m.get("heading"), "doc_id": m.get("parent_doc_id"),
            "version": m.get("version"), "status": m.get("status"), "rerank_score": h.rerank_score,
            "sent_to_llm": h.sent}


# ---------------------------------------------------------------- tools

def search_policy(ctx: ToolContext, query: str) -> ToolResult:
    """Hybrid retrieval as this user. Adds every authorised candidate to the evidence store."""
    try:
        result, attempts, errors = _with_retry(
            ctx, "search_policy", lambda: ctx.retriever.search(query, list(ctx.roles), ctx.policy_date))
    except _Failed as f:
        return _failure("search_policy", f)
    for h in result.candidates:
        if _authorised(ctx, h.meta):
            ctx.evidence.add(h.meta)
    hits = [_hit_summary(h) for h in result.hits if _authorised(ctx, h.meta)]
    data = {"query": query, "outcome": result.outcome, "hits": hits}
    if not hits:
        return ToolResult("search_policy", Code.NO_EVIDENCE, "No relevant policy section was found in the "
                          "documents available to you.", data, attempts=attempts, errors=errors)
    return ToolResult("search_policy", Code.OK, f"Found {len(hits)} relevant policy sections.", data,
                      [h["id"] for h in hits], attempts=attempts, errors=errors)


def answer_policy(ctx: ToolContext, question: str) -> ToolResult:
    """The Week 2 grounded answer: refusal gate, cited answer, citation check. Unchanged pipeline."""
    try:
        a, attempts, errors = _with_retry(
            ctx, "answer_policy", lambda: ctx.rag.answer(question, list(ctx.roles), ctx.policy_date))
    except _Failed as f:
        return _failure("answer_policy", f)
    data = {"question": question, "sources": [h.id for h in a.hits],
            "invalid_citations": a.invalid_citations,
            "outcome": a.retrieval.outcome if a.retrieval else None}
    if not a.hits:
        return ToolResult("answer_policy", Code.NO_EVIDENCE, a.text, data | {"answer": a.text},
                          attempts=attempts, errors=errors)
    if a.invalid_citations:
        # The unverified answer text is dropped here, so nothing downstream can display it by accident.
        return ToolResult("answer_policy", Code.INVALID_CITATIONS, "The answer couldn't be checked against the "
                          "policies, so it isn't shown. Please rephrase the question or contact the People team at "
                          f"{PEOPLE_TEAM}.", data | {"answer_withheld": True}, attempts=attempts, errors=errors)
    data["answer"] = a.text
    return ToolResult("answer_policy", Code.OK, a.text, data, a.citations, attempts=attempts, errors=errors)


def _gather_evidence(ctx: ToolContext, rule_ids: list[str]) -> tuple[dict[str, dict], int, list[str]]:
    """Fetch any required policy section not yet in the store, using fixed queries as this user.
    Returns only authorised chunks. Raises _Failed if retrieval keeps failing."""
    needed = {RULES[r].chunk_id for r in rule_ids}
    attempts, errors = 0, []
    for chunk_id in sorted(needed - ctx.evidence.chunks.keys()):
        result, n, errs = _with_retry(ctx, f"evidence:{chunk_id}", lambda: ctx.retriever.search(
            EVIDENCE_QUERIES[chunk_id], list(ctx.roles), ctx.policy_date))
        attempts, errors = attempts + n, errors + errs
        for h in result.candidates:
            if _authorised(ctx, h.meta):
                ctx.evidence.add(h.meta)
    # Re-check at use time too: the store only ever holds retriever output, but never trust it blindly.
    authorised = {cid: c for cid, c in ctx.evidence.chunks.items() if cid in needed and _authorised(ctx, c)}
    return authorised, attempts, errors


def _evidence_problem(tool: str, problems: list[str], attempts: int, errors: list[str]) -> ToolResult:
    return ToolResult(tool, Code.INSUFFICIENT_EVIDENCE, "The policy text needed for this decision could not be "
                      f"confirmed, so no decision was made. Contact the People team at {PEOPLE_TEAM}.",
                      data={"problems": problems}, attempts=attempts, errors=errors)


def _profile(employment_type, uk_based, working_pattern) -> EmployeeProfile:
    return EmployeeProfile(employment_type, uk_based, working_pattern)


_ELIGIBILITY_CODES = {
    ("eligible", None): Code.OK,
    ("needs_info", None): Code.MISSING_EMPLOYEE_INFO,
    ("not_eligible", CONTRACTOR_EXCLUDED): Code.NOT_ELIGIBLE,
    ("cannot_determine", EMPLOYMENT_TYPE_NOT_COVERED): Code.UNSUPPORTED_EMPLOYMENT_TYPE,
    ("cannot_determine", LOCATION_NOT_COVERED): Code.UNSUPPORTED_LOCATION,
    ("cannot_determine", EVIDENCE_UNVERIFIED): Code.INSUFFICIENT_EVIDENCE,
}


def check_eligibility_tool(ctx: ToolContext, employment_type: str | None = None, uk_based: bool | None = None,
                           working_pattern: str | None = None) -> ToolResult:
    tool = "check_eligibility"
    try:
        evidence, attempts, errors = _gather_evidence(ctx, ELIGIBILITY_RULES)
    except _Failed as f:
        return _failure(tool, f)
    ev = verify_evidence(evidence, ELIGIBILITY_RULES)
    if not ev.ok:
        return _evidence_problem(tool, ev.problems, attempts, errors)
    profile = _profile(employment_type, uk_based, working_pattern)
    r = check_eligibility(profile, evidence)
    data = {"status": r.status, "reason_id": r.reason_id, "missing": r.missing, "reasons": r.reasons,
            "notes": r.notes, "profile": asdict(profile),
            "profile_source": "self-declared by the employee; not verified"}
    # Explicit (status, reason_id) -> code. Anything unmapped is a defect: KeyError reaches the safety net.
    code = _ELIGIBILITY_CODES[(r.status, r.reason_id)]
    message = {Code.OK: "Eligible under the annual leave policy.",
               Code.NOT_ELIGIBLE: "Not covered by the annual leave policy.",
               Code.MISSING_EMPLOYEE_INFO: "More information is needed: " + ", ".join(r.missing) + "."}.get(
        code, " ".join(r.reasons))
    return ToolResult(tool, code, message, data, r.citations, attempts=attempts, errors=errors)


def calculate_dates(ctx: ToolContext, start_date: str, end_date: str) -> ToolResult:
    tool = "calculate_dates"
    try:
        start, end = parse_date(start_date), parse_date(end_date)
    except ValueError as e:
        return ToolResult(tool, Code.INVALID_DATES, str(e))
    try:
        evidence, attempts, errors = _gather_evidence(ctx, DATE_RULES)
    except _Failed as f:
        return _failure(tool, f)
    ev = verify_evidence(evidence, DATE_RULES)
    if not ev.ok:
        return _evidence_problem(tool, ev.problems, attempts, errors)
    a = assess_dates(start, end, ctx.today, ctx.holidays, evidence)
    flags = list(a.flags)
    until = min((c["effective_until"] for c in evidence.values()), default=None)
    if until is not None and int(end.strftime("%Y%m%d")) > until:
        flags.append(f"The policy version used ends on {until} before your leave ends; the rules may change.")
    data = {"status": a.status, "request_date": ctx.today.isoformat(), "start": start.isoformat(),
            "end": end.isoformat(), "reasons": a.reasons, "flags": flags,
            "first_leave_day": a.first_leave_day and a.first_leave_day.isoformat(),
            "leave_days": [d.isoformat() for d in a.leave_days], "working_days_requested": a.working_day_count,
            "working_days_note": BALANCE_DISCLAIMER,
            "latest_request_date": a.latest_request_date and a.latest_request_date.isoformat(),
            "earliest_valid_start": a.earliest_valid_start and a.earliest_valid_start.isoformat(),
            "approvers": a.approvers}
    code = {"ok": Code.OK, "needs_change": Code.NOTICE_TOO_SHORT, "invalid": Code.INVALID_DATES}.get(
        a.status, Code.INSUFFICIENT_EVIDENCE)
    message = {Code.OK: f"{a.working_day_count} working days requested, with enough notice.",
               Code.NOTICE_TOO_SHORT: " ".join(a.reasons)}.get(code, " ".join(a.reasons))
    assumptions = a.assumptions or [WORKING_DAY_ASSUMPTION]
    return ToolResult(tool, code, message, data, a.citations, assumptions, attempts, errors)


def prepare_draft(ctx: ToolContext, employment_type: str | None, uk_based: bool | None, working_pattern: str | None,
                  start_date: str, end_date: str, employee_note: str = "") -> ToolResult:
    """Build the draft in memory. Re-runs both checks itself rather than trusting earlier results.
    Nothing is saved: saving happens only after the employee approves, in a later stage.

    Checks run in order and stop at the first that fails: if eligibility fails, dates are not checked.
    A refusal is PRECONDITION_FAILED with data["reason_code"] set to the failing check's own code (e.g.
    NOT_ELIGIBLE, NOTICE_TOO_SHORT), so the graph explains the real reason. A fault stays TOOL_FAILURE."""
    tool = "prepare_draft"
    checks = {}
    for name, run in (("eligibility", lambda: check_eligibility_tool(ctx, employment_type, uk_based,
                                                                     working_pattern)),
                      ("dates", lambda: calculate_dates(ctx, start_date, end_date))):
        r = checks[name] = run()
        if r.code == Code.TOOL_FAILURE:
            return ToolResult(tool, Code.TOOL_FAILURE, r.message, r.data | {"blocked_by": name},
                              attempts=r.attempts, errors=r.errors)
        if not r.ok:
            return ToolResult(tool, Code.PRECONDITION_FAILED, r.message,
                              data={"blocked_by": name, "reason_code": r.code.value, name: r.to_dict()},
                              citations=r.citations, assumptions=r.assumptions,
                              attempts=sum(c.attempts for c in checks.values()),
                              errors=[e for c in checks.values() for e in c.errors])
    elig, dates = checks["eligibility"], checks["dates"]
    d = dates.data
    policy = next(c for c in ctx.evidence.chunks.values() if c["_id"] == RULES["notice"].chunk_id)
    # Each approval is looked up by its role ID and carries the rule that requires it; order is irrelevant.
    approvals = []
    for role_id in d["approvers"]:
        role = APPROVER_ROLES[role_id]  # an unknown role is a defect: KeyError reaches the safety net
        rule = RULES[role.rule_id]
        approvals.append({"role": role.role_id, "approver": role.label, "rule_id": role.rule_id,
                          "basis": rule.quote, "citation": rule.chunk_id, "status": "not requested yet"})
    citations = list(dict.fromkeys(elig.citations + dates.citations))
    draft = {
        "draft_id": ctx.new_id(), "draft_version": 1, "leave_type": "annual_leave",
        "status": DRAFT_STATUS, "status_text": DRAFT_STATUS_TEXT,
        "prepared_on": ctx.today.isoformat(),
        "employee": {"reference": ctx.employee_ref, **elig.data["profile"],
                     "details_source": elig.data["profile_source"]},
        "dates": {"start": d["start"], "end": d["end"], "first_leave_day": d["first_leave_day"],
                  "leave_days": d["leave_days"], "working_days_requested": d["working_days_requested"],
                  "working_days_note": BALANCE_DISCLAIMER},
        "notice": {"request_date": d["request_date"], "latest_request_date": d["latest_request_date"],
                   "rule": RULES["notice"].quote, "citation": RULES["notice"].chunk_id},
        "approvals_required": approvals,
        "eligibility_notes": elig.data["notes"],
        "flags": d["flags"],
        "assumptions": dates.assumptions,
        "next_steps": [RULES["notice"].quote, RULES["manager_decision"].quote, RULES["not_approved"].quote],
        "policy": {"doc_id": POLICY_DOC_ID, "version": policy["version"], "status": policy["status"],
                   "effective_from": policy["effective_from"], "policies_in_force_on": ctx.policy_date.isoformat()},
        "citations": citations,
        "employee_note": employee_note,
    }
    return ToolResult(tool, Code.OK, "Draft prepared for your review. Nothing has been saved or submitted.",
                      {"draft": draft}, citations, dates.assumptions, elig.attempts + dates.attempts,
                      elig.errors + dates.errors)


# ---------------------------------------------------------------- dispatcher

_EMPLOYMENT = {"type": ["string", "null"], "enum": ["permanent", "contractor", "other", None],
               "description": "As stated by the employee. 'other' for fixed-term, agency, intern, etc."}
_UK = {"type": ["boolean", "null"], "description": "Whether the employee says they are UK-based."}
_PATTERN = {"type": ["string", "null"], "enum": ["full_time", "part_time", None]}
_DATE = {"type": "string", "maxLength": 20,
         "description": DATE_FORMAT_HINT + " (YYYY-MM-DD also accepted), as confirmed with the employee."}

# JSON-schema style specs: run_tool validates against these now; Stage 4 hands them to the LLM.
TOOL_SPECS = {
    "search_policy": {"fn": search_policy, "description": "Search the HR policies the employee may see.",
                      "params": {"query": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT}}, "required": ["query"]},
    "answer_policy": {"fn": answer_policy, "description": "Answer a general HR policy question with citations.",
                      "params": {"question": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT}}, "required": ["question"]},
    "check_eligibility": {"fn": check_eligibility_tool, "description": "Check annual leave eligibility.",
                          "params": {"employment_type": _EMPLOYMENT, "uk_based": _UK, "working_pattern": _PATTERN},
                          "required": []},
    "calculate_dates": {"fn": calculate_dates, "description": "Count working days and check notice and approvers.",
                        "params": {"start_date": _DATE, "end_date": _DATE}, "required": ["start_date", "end_date"]},
    "prepare_draft": {"fn": prepare_draft, "description": "Prepare a draft annual leave request for review.",
                      "params": {"employment_type": _EMPLOYMENT, "uk_based": _UK, "working_pattern": _PATTERN,
                                 "start_date": _DATE, "end_date": _DATE,
                                 "employee_note": {"type": "string", "maxLength": MAX_NOTE}},
                      "required": ["employment_type", "uk_based", "working_pattern", "start_date", "end_date"]},
}
_PY_TYPES = {"string": str, "boolean": bool, "null": type(None)}


def _validate(spec: dict, args: dict) -> list[str]:
    problems = [f"unexpected argument {k!r}: tools cannot set roles, policy metadata or evidence"
                for k in args if k not in spec["params"]]
    problems += [f"missing argument {k!r}" for k in spec["required"] if k not in args]
    for k, v in args.items():
        p = spec["params"].get(k)
        if p is None:
            continue
        types = p["type"] if isinstance(p["type"], list) else [p["type"]]
        if not any(isinstance(v, _PY_TYPES[t]) and not (t == "string" and isinstance(v, bool)) for t in types):
            problems.append(f"{k!r} must be {' or '.join(types)}")
        elif "enum" in p and v not in p["enum"]:
            problems.append(f"{k!r} must be one of {[e for e in p['enum'] if e is not None]}")
        elif isinstance(v, str) and len(v) > p.get("maxLength", 10_000):
            problems.append(f"{k!r} is longer than {p['maxLength']} characters")
        elif isinstance(v, str) and len(v.strip()) < p.get("minLength", 0):
            problems.append(f"{k!r} must not be blank")
    return problems


def run_tool(ctx: ToolContext, name: str, args: dict) -> ToolResult:
    """The single entry point the graph uses. Validates arguments, then calls the tool.
    Last-resort safety net: any unexpected exception becomes TOOL_FAILURE with failure="internal", never a
    business outcome. Its details go only to the restricted diagnostic log, under the reference returned."""
    try:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            return ToolResult(name, Code.UNKNOWN_TOOL, f"There is no tool called {name!r}.")
        if not isinstance(args, dict):
            return ToolResult(name, Code.INVALID_ARGUMENTS, "Arguments must be an object.")
        problems = _validate(spec, args)
        if problems:
            return ToolResult(name, Code.INVALID_ARGUMENTS, "; ".join(problems), data={"problems": problems})
        return spec["fn"](ctx, **args)
    except Exception as e:
        ref = diagnostics.new_ref()
        diagnostics.record(ref, "internal", e, tool=str(name)[:50])
        return ToolResult(str(name)[:50], Code.TOOL_FAILURE,
                          ERROR_MESSAGES["unavailable"].format(people_team=PEOPLE_TEAM, ref=ref),
                          data={"failure": "internal", "retryable": False, "reference": ref},
                          errors=[f"internal {type(e).__name__} (diagnostic ref {ref})"])
