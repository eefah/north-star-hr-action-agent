"""Annual leave rules, applied in plain Python so the LLM can't change the outcome.

Every rule is tied to a verbatim sentence from the annual leave policy and the chunk ID it lives in.
A rule is only applied when the retrieved evidence still contains that sentence in a current v2.0
policy (verify_evidence). If the wording, version or status changes, the agent stops and refers the
employee to the People team instead of applying a stale rule. Nothing here calculates leave balances
or part-time entitlement: the policy gives neither.
"""
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from agent.dates import WORKING_DAY_ASSUMPTION, BankHolidays, CalendarError, non_working_days, \
    shift_working_days, working_days, is_working_day

POLICY_DOC_ID = "NS-ANNUAL-LEAVE"
POLICY_VERSION = "2.0"
PEOPLE_TEAM = "people@northstar-demo.example"  # from annual_leave#contact


@dataclass(frozen=True)
class PolicyRule:
    chunk_id: str
    quote: str  # verbatim from the policy; checked against the retrieved chunk before use


RULES = {
    "eligible_group": PolicyRule("annual_leave#eligibility", "Permanent UK employees receive 25 days of paid "
                                 "annual leave per calendar year, plus England and Wales bank holidays."),
    "part_time": PolicyRule("annual_leave#eligibility", "Part-time entitlement is pro-rated by contracted weekly hours."),
    "contractors": PolicyRule("annual_leave#eligibility", "Contractors are outside this policy."),
    "leave_year": PolicyRule("annual_leave#eligibility", "The leave year runs from 1 January to 31 December."),
    "notice": PolicyRule("annual_leave#how-to-request", "Submit requests in PeopleHub at least 10 working days "
                         "before the first day of leave."),
    "manager_decision": PolicyRule("annual_leave#how-to-request", "Your line manager approves or declines within "
                                   "3 working days."),
    "not_approved": PolicyRule("annual_leave#how-to-request", "Do not treat a submitted request as approved."),
    "long_leave": PolicyRule("annual_leave#how-to-request", "Requests for more than 10 consecutive working days "
                             "also need approval from Priya Shah, Head of People."),
    "no_balances": PolicyRule("annual_leave#contact", "This policy contains no live individual leave balances."),
}
NOTICE_WORKING_DAYS = 10  # RULES["notice"]
LONG_LEAVE_MAX_WORKING_DAYS = 10  # RULES["long_leave"]: more than this needs the Head of People too


@dataclass(frozen=True)
class ApproverRole:
    """A named approver, tied to the rule that requires them. Approvers are referred to by role_id only,
    never by their position in a list."""
    role_id: str
    label: str
    rule_id: str  # key in RULES: the policy sentence that makes this approval necessary


LINE_MANAGER = "line_manager"
HEAD_OF_PEOPLE = "head_of_people"
APPROVER_ROLES = {
    LINE_MANAGER: ApproverRole(LINE_MANAGER, "Line manager", "manager_decision"),
    HEAD_OF_PEOPLE: ApproverRole(HEAD_OF_PEOPLE, "Priya Shah, Head of People", "long_leave"),
}

ELIGIBILITY_RULES = ["eligible_group", "part_time", "contractors", "no_balances"]
DATE_RULES = ["notice", "manager_decision", "not_approved", "long_leave", "leave_year"]

# Interpretations the policy leaves open. Shown to the employee whenever they affect a result.
NOTICE_ASSUMPTION = (
    "Assumption, not policy: 'at least 10 working days before the first day of leave' is counted as: the request "
    "date must be on or before the working day that falls 10 working days before the first day of leave."
)
CONSECUTIVE_ASSUMPTION = (
    "Assumption, not policy: all working days in one continuous date range count as consecutive, even when a "
    "weekend or bank holiday falls inside it."
)
BANK_HOLIDAY_ASSUMPTION = (
    "Assumption, not policy: bank holidays inside the range are not counted as annual leave days, because the "
    "policy grants bank holidays in addition to the 25 days."
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class EvidenceCheck:
    ok: bool
    problems: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


def verify_evidence(evidence: dict[str, dict], rule_ids: list[str]) -> EvidenceCheck:
    """Check the retrieved chunks still support each rule. evidence maps chunk ID -> chunk record
    (chunk_text, parent_doc_id, version, status), as produced by ingest.build_chunks and held in Hit.meta."""
    problems, citations = [], []
    for rid in rule_ids:
        rule = RULES[rid]
        chunk = evidence.get(rule.chunk_id)
        if chunk is None:
            problems.append(f"{rule.chunk_id} was not retrieved")
        elif chunk.get("parent_doc_id") != POLICY_DOC_ID or chunk.get("version") != POLICY_VERSION:
            problems.append(f"{rule.chunk_id} is {chunk.get('parent_doc_id')} v{chunk.get('version')}; the rules "
                            f"were written for {POLICY_DOC_ID} v{POLICY_VERSION}")
        elif chunk.get("status") != "current":
            problems.append(f"{rule.chunk_id} has status {chunk.get('status')!r}, not 'current'")
        elif _normalise(rule.quote) not in _normalise(chunk.get("chunk_text", "")):
            problems.append(f"{rule.chunk_id} no longer contains: \"{rule.quote}\"")
        elif rule.chunk_id not in citations:
            citations.append(rule.chunk_id)
    return EvidenceCheck(not problems, list(dict.fromkeys(problems)), citations)


@dataclass
class EmployeeProfile:
    """What the employee tells the agent. Synthetic in tests and the demo."""
    employment_type: str | None = None  # "permanent", "contractor" or "other"
    uk_based: bool | None = None
    working_pattern: str | None = None  # "full_time" or "part_time"


# Explicit reasons for not_eligible / cannot_determine, so callers never infer them from the profile.
EVIDENCE_UNVERIFIED = "evidence_unverified"
CONTRACTOR_EXCLUDED = "contractor_excluded"  # RULES["contractors"]
EMPLOYMENT_TYPE_NOT_COVERED = "employment_type_not_covered"
LOCATION_NOT_COVERED = "location_not_covered"
ELIGIBILITY_REASONS = {EVIDENCE_UNVERIFIED, CONTRACTOR_EXCLUDED, EMPLOYMENT_TYPE_NOT_COVERED, LOCATION_NOT_COVERED}


@dataclass
class EligibilityResult:
    status: str  # "eligible", "not_eligible", "needs_info" or "cannot_determine"
    reason_id: str | None = None  # why, for not_eligible and cannot_determine: one of ELIGIBILITY_REASONS
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


def check_eligibility(profile: EmployeeProfile, evidence: dict[str, dict]) -> EligibilityResult:
    ev = verify_evidence(evidence, ELIGIBILITY_RULES)
    if not ev.ok:
        return EligibilityResult("cannot_determine", EVIDENCE_UNVERIFIED,
                                 ev.problems + [f"Contact the People team at {PEOPLE_TEAM}."])
    cite = ev.citations
    if profile.employment_type is None:
        missing = ["employment_type"] + [f for f in ("uk_based", "working_pattern") if getattr(profile, f) is None]
        return EligibilityResult("needs_info", missing=missing, citations=cite)
    if profile.employment_type == "contractor":
        return EligibilityResult("not_eligible", CONTRACTOR_EXCLUDED, [RULES["contractors"].quote,
                                                  f"Contact the People team at {PEOPLE_TEAM}."], citations=cite)
    if profile.employment_type != "permanent":
        return EligibilityResult("cannot_determine", EMPLOYMENT_TYPE_NOT_COVERED, [
            "The policy covers permanent UK employees and excludes contractors. It says nothing about other "
            f"employment types, so this needs the People team at {PEOPLE_TEAM}."], citations=cite)
    missing = [f for f in ("uk_based", "working_pattern") if getattr(profile, f) is None]
    if missing:
        return EligibilityResult("needs_info", missing=missing, citations=cite)
    if not profile.uk_based:
        return EligibilityResult("cannot_determine", LOCATION_NOT_COVERED, [
            "The policy covers permanent UK employees. It doesn't say how it applies outside the UK, so this needs "
            f"the People team at {PEOPLE_TEAM}."], citations=cite)
    notes = [RULES["eligible_group"].quote] if profile.working_pattern == "full_time" else [
        RULES["part_time"].quote + " The policy gives no formula, so the agent does not calculate your entitlement."]
    notes.append("Your remaining balance isn't in the policy, so the agent can't check it. Your line manager "
                 "checks it when deciding the request.")
    return EligibilityResult("eligible", None, ["Permanent UK employees are covered by the annual leave policy."],
                             notes=notes, citations=cite)


@dataclass
class DateAssessment:
    status: str  # "ok", "needs_change" (notice too short), "invalid" or "cannot_determine"
    start: date | None = None
    end: date | None = None
    leave_days: list[date] = field(default_factory=list)  # working days in the range
    first_leave_day: date | None = None
    latest_request_date: date | None = None
    earliest_valid_start: date | None = None
    approvers: list[str] = field(default_factory=list)  # role IDs, keys of APPROVER_ROLES
    flags: list[str] = field(default_factory=list)  # things the policy doesn't settle, for the manager to confirm
    assumptions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    @property
    def working_day_count(self) -> int:
        return len(self.leave_days)


def earliest_compliant_start(request_date: date, holidays: BankHolidays) -> date:
    """The first working day whose notice deadline is on or after request_date."""
    d = request_date + timedelta(1)
    while not (is_working_day(d, holidays) and
               shift_working_days(d, -NOTICE_WORKING_DAYS, holidays) >= request_date):
        d += timedelta(1)
    return d


def assess_dates(start: date, end: date, request_date: date, holidays: BankHolidays,
                 evidence: dict[str, dict]) -> DateAssessment:
    ev = verify_evidence(evidence, DATE_RULES)
    if not ev.ok:
        return DateAssessment("cannot_determine", start, end,
                              reasons=ev.problems + [f"Contact the People team at {PEOPLE_TEAM}."])
    if end < start:
        return DateAssessment("invalid", start, end, reasons=["The end date is before the start date."])
    if start < request_date:
        return DateAssessment("invalid", start, end, reasons=["The leave starts before today."])
    try:
        leave_days = working_days(start, end, holidays)
        if not leave_days:
            return DateAssessment("invalid", start, end, reasons=["There are no working days in this range."],
                                  assumptions=[WORKING_DAY_ASSUMPTION])
        first = leave_days[0]
        latest = shift_working_days(first, -NOTICE_WORKING_DAYS, holidays)
        earliest = earliest_compliant_start(request_date, holidays)
        days_off = non_working_days(start, end, holidays)
    except CalendarError as e:
        return DateAssessment("invalid", start, end, reasons=[f"Can't count working days: {e}."])

    a = DateAssessment("ok", start, end, leave_days, first, latest, earliest, [LINE_MANAGER],
                       assumptions=[WORKING_DAY_ASSUMPTION, NOTICE_ASSUMPTION], citations=ev.citations)
    if request_date > latest:
        a.status = "needs_change"
        a.reasons.append(f"{RULES['notice'].quote} For a first day of leave on {first.isoformat()}, the request "
                         f"had to be made by {latest.isoformat()}. The earliest first day of leave that gives "
                         f"enough notice is {earliest.isoformat()}.")
    if len(leave_days) > LONG_LEAVE_MAX_WORKING_DAYS:
        a.approvers.append(HEAD_OF_PEOPLE)
        a.reasons.append(RULES["long_leave"].quote)
        a.assumptions.append(CONSECUTIVE_ASSUMPTION)
    holidays_inside = [d for d in days_off if d in holidays.dates]
    if holidays_inside:
        a.assumptions.append(BANK_HOLIDAY_ASSUMPTION)
        a.flags.append("Bank holidays inside the range, not counted as leave days: "
                       + ", ".join(d.isoformat() for d in holidays_inside) + ".")
    if first != start:
        a.flags.append(f"{start.isoformat()} is not a working day, so the first day of leave is "
                       f"{first.isoformat()}.")
    if start.year != end.year:
        a.flags.append(f"{RULES['leave_year'].quote} This request spans two leave years. The policy doesn't say "
                       "how such a request is split, so confirm it with your line manager.")
    return a
