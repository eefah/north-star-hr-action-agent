"""Turn a tool result into one of three outcomes and an explanation the employee can understand.

    can_proceed       evidence exists, the rules are satisfied, the next authorised step is available
    cannot_proceed    a known rule (or invalid dates) prevents the action; the specific reason is given
    cannot_determine  evidence is missing or a technical fault prevents a reliable decision. Never
                      described as ineligibility; always gives a recovery route.

The explanation is built from the underlying result code and its data, never from a generic message.
A refused draft (PRECONDITION_FAILED) is explained by the check that actually failed. Fault messages are
the approved ERROR_MESSAGES already set by the tools; no technical detail is added here.
"""
from agent.rules import PEOPLE_TEAM, RULES
from agent.tools import Code

CAN_PROCEED = "can_proceed"
CANNOT_PROCEED = "cannot_proceed"
CANNOT_DETERMINE = "cannot_determine"

CATEGORY = {
    Code.OK: CAN_PROCEED,
    Code.NOT_ELIGIBLE: CANNOT_PROCEED,
    Code.NOTICE_TOO_SHORT: CANNOT_PROCEED,
    Code.INVALID_DATES: CANNOT_PROCEED,
    Code.MISSING_EMPLOYEE_INFO: CANNOT_DETERMINE,
    Code.UNSUPPORTED_EMPLOYMENT_TYPE: CANNOT_DETERMINE,
    Code.UNSUPPORTED_LOCATION: CANNOT_DETERMINE,
    Code.INSUFFICIENT_EVIDENCE: CANNOT_DETERMINE,
    Code.NO_EVIDENCE: CANNOT_DETERMINE,
    Code.INVALID_CITATIONS: CANNOT_DETERMINE,
    Code.TOOL_FAILURE: CANNOT_DETERMINE,
    Code.INVALID_ARGUMENTS: CANNOT_DETERMINE,
    Code.UNKNOWN_TOOL: CANNOT_DETERMINE,
    Code.PRECONDITION_FAILED: CANNOT_DETERMINE,  # only when no underlying reason is available
}
NOT_A_DECISION = "This is not a decision about your leave."
CONTACT = f"the People team at {PEOPLE_TEAM}"
FIELD_NAMES = {"employment_type": "your employment type (permanent, contractor or other)",
               "uk_based": "whether you are based in the UK",
               "working_pattern": "whether you work full time or part time"}


def underlying(result: dict) -> dict:
    """For a refused draft, the result of the check that failed; otherwise the result itself."""
    if result.get("code") == Code.PRECONDITION_FAILED.value:
        inner = result.get("data", {}).get(result.get("data", {}).get("blocked_by", ""))
        if isinstance(inner, dict) and "code" in inner:
            return inner
    return result


def _quote(rule_id: str) -> str:
    rule = RULES[rule_id]
    return f"“{rule.quote}” [{rule.chunk_id}]"


def explain(result: dict) -> dict:
    """result is ToolResult.to_dict(). Returns the outcome stored in graph state and shown to the employee."""
    r = underlying(result)
    code = Code(r["code"])
    data = r.get("data", {})
    category = CATEGORY[code]  # an unmapped code is a defect: KeyError reaches the graph's safety net
    recovery, retryable = None, False

    if code == Code.NOT_ELIGIBLE:
        message = (f"This request can't go ahead under the annual leave policy, which says {_quote('contractors')}. "
                   f"If your details are wrong, or to ask which leave arrangements apply to you, contact {CONTACT}.")
        recovery = f"Contact {CONTACT}."
    elif code == Code.NOTICE_TOO_SHORT:
        message = (f"This request doesn't give enough notice. The policy says {_quote('notice')}. "
                   f"For a first day of leave on {data.get('first_leave_day')}, the request had to be made by "
                   f"{data.get('latest_request_date')}. The earliest first day of leave that gives enough "
                   f"notice is {data.get('earliest_valid_start')}.")
        recovery = f"Start a new request with a first day of leave on or after {data.get('earliest_valid_start')}."
    elif code == Code.INVALID_DATES:
        message = f"These dates can't be used: {r.get('message')} Nothing has been decided about your leave."
        recovery = "Start a new request with different dates."
    elif code in (Code.UNSUPPORTED_EMPLOYMENT_TYPE, Code.UNSUPPORTED_LOCATION):
        who = "your employment type" if code == Code.UNSUPPORTED_EMPLOYMENT_TYPE else "employees based outside the UK"
        message = (f"The annual leave policy covers permanent UK employees and doesn't say how it applies to {who}, "
                   f"so the agent can't decide this. {NOT_A_DECISION} Please contact {CONTACT}.")
        recovery = f"Contact {CONTACT}."
    elif code == Code.MISSING_EMPLOYEE_INFO:
        fields = [FIELD_NAMES.get(f, f) for f in data.get("missing", [])]
        message = (f"The agent needs {', '.join(fields) or 'more details'} to check eligibility and didn't receive "
                   f"them, so nothing has been decided. {NOT_A_DECISION}")
        recovery = "Start a new request and include these details."
    elif code == Code.INSUFFICIENT_EVIDENCE:
        message = (f"The agent couldn't confirm the current policy wording it needs, so it hasn't made a decision. "
                   f"{NOT_A_DECISION} Please contact {CONTACT}.")
        recovery = f"Contact {CONTACT}."
    elif code == Code.NO_EVIDENCE:
        message = (f"None of the policies available to you cover this, so the agent can't answer it. "
                   f"Please contact {CONTACT}.")
        recovery = f"Contact {CONTACT}, or rephrase the question."
    elif code == Code.INVALID_CITATIONS:
        message = r["message"]  # the answer itself is withheld by the tool
        recovery = f"Rephrase the question or contact {CONTACT}."
    elif code == Code.TOOL_FAILURE:
        message = r["message"]  # one of the approved ERROR_MESSAGES
        retryable = bool(data.get("retryable"))
        recovery = "Try again in a few minutes." if retryable else f"Contact {CONTACT}."
    elif code in (Code.INVALID_ARGUMENTS, Code.UNKNOWN_TOOL, Code.PRECONDITION_FAILED):
        message = ("Some of the details weren't in a form the agent can use, so nothing has been decided. "
                   f"{NOT_A_DECISION}")
        recovery = "Check the details and start a new request."
    else:  # Code.OK
        message = r["message"]
    return {"category": category, "code": code.value, "message": message,
            "citations": list(r.get("citations", [])), "recovery": recovery, "retryable": retryable}
