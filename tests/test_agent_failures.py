"""Failure handling, refusal reasons, approver roles and input checks in agent/tools.py. Offline; synthetic data.

    python -m unittest discover tests
"""
import json
import logging
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import groq
import httpx
from pinecone.errors import PineconeTimeoutError, PineconeValueError, RateLimitError, ServiceError, \
    UnauthorizedError

from agent import diagnostics, tools
from agent.rules import APPROVER_ROLES, HEAD_OF_PEOPLE, LINE_MANAGER, RULES, EligibilityResult
from agent.tools import ERROR_MESSAGES, EVIDENCE_QUERIES, Code, classify_error, run_tool
from tests.fakes import FakeLLM, make_ctx

FULL_TIME = {"employment_type": "permanent", "uk_based": True, "working_pattern": "full_time"}
DATES = {"start_date": "23/11/2026", "end_date": "25/11/2026"}
SEARCH = ("search_policy", {"query": "annual leave notice"})


def _groq_status(cls, code):
    return cls("x", response=httpx.Response(code, request=httpx.Request("POST", "https://api.groq.test")), body=None)


def _raising(exc):
    def fn(*args, **kwargs):
        raise exc
    return fn


def _approved(message):
    return any(message == m or message.startswith(m.split("{")[0]) for m in ERROR_MESSAGES.values())


class ClassificationTests(unittest.TestCase):
    """Real SDK exception classes, so a change in either SDK shows up here."""
    def test_errors_likely_to_recover_are_transient(self):
        req = httpx.Request("POST", "https://api.groq.test")
        for e in (TimeoutError(), ConnectionError(), PineconeTimeoutError("timed out"), ServiceError(), RateLimitError(),
                  ServiceError("unavailable", 503), groq.APITimeoutError(req), groq.APIConnectionError(request=req),
                  _groq_status(groq.RateLimitError, 429), _groq_status(groq.InternalServerError, 502),
                  httpx.ReadTimeout("t")):
            with self.subTest(e=type(e).__name__):
                self.assertEqual(classify_error(e), "transient")

    def test_rejected_requests_are_not_retried(self):
        for e in (UnauthorizedError(), _groq_status(groq.AuthenticationError, 401),
                  _groq_status(groq.BadRequestError, 400), _groq_status(groq.PermissionDeniedError, 403),
                  PineconeValueError("bad top_k")):
            with self.subTest(e=type(e).__name__):
                self.assertEqual(classify_error(e), "rejected")

    def test_code_defects_are_not_service_errors(self):
        for e in (TypeError(), KeyError("x"), AttributeError(), ValueError(), ZeroDivisionError()):
            with self.subTest(e=type(e).__name__):
                self.assertEqual(classify_error(e), "defect")


class RetryPolicyTests(unittest.TestCase):
    def run_failing(self, exc, name="search_policy", args=None):
        ctx, sleeps = make_ctx()
        ctx.retriever.search = _raising(exc)
        return run_tool(ctx, name, args or SEARCH[1]), sleeps

    def test_invalid_authentication_stops_at_once(self):
        r, sleeps = self.run_failing(UnauthorizedError())
        self.assertEqual((r.code, r.attempts, sleeps), (Code.TOOL_FAILURE, 1, []))
        self.assertEqual((r.data["failure"], r.data["retryable"]), ("rejected", False))

    def test_malformed_request_to_the_llm_stops_at_once(self):
        llm = FakeLLM("unused")
        llm.chat.completions.create = _raising(_groq_status(groq.BadRequestError, 400))
        ctx, sleeps = make_ctx(llm=llm)
        r = run_tool(ctx, "answer_policy", {"question": "How many annual leave days do permanent UK employees get?"})
        self.assertEqual((r.code, r.attempts, sleeps, r.data["failure"]), (Code.TOOL_FAILURE, 1, [], "rejected"))

    def test_rate_limits_and_outages_are_retried(self):
        for exc in (RateLimitError(), ServiceError("unavailable", 503), PineconeTimeoutError("timed out")):
            with self.subTest(e=type(exc).__name__):
                r, sleeps = self.run_failing(exc)
                self.assertEqual((r.code, r.attempts, sleeps), (Code.TOOL_FAILURE, 3, [0.5, 1.5]))
                self.assertEqual((r.data["failure"], r.data["retryable"]), ("transient", True))
                self.assertEqual(r.message, ERROR_MESSAGES["transient"])

    def test_code_defect_is_not_retried_and_goes_to_the_safety_net(self):
        r, sleeps = self.run_failing(TypeError("unsupported operand"))
        self.assertEqual((r.code, r.attempts, sleeps), (Code.TOOL_FAILURE, 0, []))
        self.assertEqual((r.data["failure"], r.data["retryable"]), ("internal", False))


class SafetyNetTests(unittest.TestCase):
    def test_unexpected_error_is_a_fault_not_a_refusal(self):
        for name, args in (("check_eligibility", FULL_TIME), ("prepare_draft", FULL_TIME | DATES)):
            with self.subTest(tool=name), mock.patch("agent.tools.check_eligibility", _raising(KeyError("boom"))):
                r = run_tool(make_ctx()[0], name, args)
                self.assertEqual((r.code, r.data["failure"]), (Code.TOOL_FAILURE, "internal"))
                self.assertNotIn(r.code, {Code.NOT_ELIGIBLE, Code.PRECONDITION_FAILED, Code.INSUFFICIENT_EVIDENCE})
                self.assertTrue(_approved(r.message))
                self.assertIn(r.data["reference"], r.message)

    def test_unmapped_eligibility_reason_is_a_defect_not_a_guess(self):
        with mock.patch("agent.tools.check_eligibility", lambda *a: EligibilityResult("cannot_determine", "new")):
            r = run_tool(make_ctx()[0], "check_eligibility", FULL_TIME)
        self.assertEqual((r.code, r.data["failure"]), (Code.TOOL_FAILURE, "internal"))

    def test_safety_net_result_is_serialisable(self):
        with mock.patch("agent.tools.check_eligibility", _raising(RuntimeError("x"))):
            json.dumps(run_tool(make_ctx()[0], "check_eligibility", FULL_TIME).to_dict())


class PrivacyTests(unittest.TestCase):
    SECRET = "pcsk_SECRETabcdef123456"
    EMAIL = "jane.doe@example.com"

    def failing_result(self):
        ctx, _ = make_ctx()
        ctx.retriever.search = _raising(UnauthorizedError(f"api_key={self.SECRET} for {self.EMAIL} Bearer tok123"))
        with self.assertLogs("northstar.diagnostics", logging.ERROR) as logs:
            r = run_tool(ctx, *SEARCH)
        return r, "\n".join(logs.output)

    def test_trace_holds_no_exception_text(self):
        r, _ = self.failing_result()
        trace = json.dumps(r.to_dict())
        for secret in (self.SECRET, self.EMAIL, "tok123", "api_key"):
            self.assertNotIn(secret, trace)
        self.assertIn("UnauthorizedError", r.errors[0])
        self.assertIn(r.data["reference"], r.errors[0])

    def test_employee_sees_only_an_approved_message(self):
        r, _ = self.failing_result()
        self.assertTrue(_approved(r.message))
        self.assertNotIn("Unauthorized", r.message)

    def test_diagnostic_log_has_details_with_secrets_removed(self):
        r, log = self.failing_result()
        self.assertIn(f"ref={r.data['reference']}", log)
        self.assertIn("Traceback", log)
        for secret in (self.SECRET, self.EMAIL, "tok123"):
            self.assertNotIn(secret, log)
        self.assertIn(diagnostics.REDACTED, log)

    def test_secret_from_the_environment_is_redacted(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "plain-looking-key-value-42"}):
            self.assertNotIn("plain-looking-key-value-42", diagnostics.redact("key plain-looking-key-value-42 used"))

    def test_diagnostic_log_never_reaches_general_logs(self):
        self.assertFalse(diagnostics.LOGGER.propagate)

    def test_log_file_is_owner_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diag" / "diagnostics.log"
            handler = diagnostics.configure(path)
            try:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            finally:
                diagnostics.LOGGER.removeHandler(handler)
                handler.close()


class RefusalReasonTests(unittest.TestCase):
    """A refused draft says why, and stops at the first failed check."""
    def test_eligibility_failure_stops_before_the_date_check(self):
        ctx, _ = make_ctx()
        r = run_tool(ctx, "prepare_draft", FULL_TIME | DATES | {"employment_type": "contractor"})
        self.assertEqual((r.code, r.reason, r.data["blocked_by"]), (Code.PRECONDITION_FAILED, Code.NOT_ELIGIBLE,
                                                                    "eligibility"))
        self.assertNotIn("dates", r.data)
        self.assertNotIn(EVIDENCE_QUERIES["annual_leave#how-to-request"], [q for q, _ in ctx.retriever.calls])

    def test_notice_too_short_is_the_reason_given(self):
        r = run_tool(make_ctx()[0], "prepare_draft", FULL_TIME | {"start_date": "10/11/2026",
                                                                  "end_date": "12/11/2026"})
        self.assertEqual((r.code, r.reason, r.data["blocked_by"]), (Code.PRECONDITION_FAILED, Code.NOTICE_TOO_SHORT,
                                                                    "dates"))
        self.assertIn(RULES["notice"].quote, r.message)
        self.assertIn("2026-11-17", r.message)  # the earliest start that gives enough notice

    def test_tool_failure_in_a_check_stays_a_tool_failure(self):
        ctx, _ = make_ctx()
        ctx.retriever.search = _raising(ConnectionError())
        r = run_tool(ctx, "prepare_draft", FULL_TIME | DATES)
        self.assertEqual((r.code, r.reason, r.data["blocked_by"]), (Code.TOOL_FAILURE, Code.TOOL_FAILURE,
                                                                    "eligibility"))

    def test_reason_of_other_results_is_their_own_code(self):
        r = run_tool(make_ctx()[0], "check_eligibility", {"employment_type": "contractor"})
        self.assertEqual(r.reason, Code.NOT_ELIGIBLE)


class ApproverRoleTests(unittest.TestCase):
    def test_each_approval_names_its_role_and_rule(self):
        r = run_tool(make_ctx()[0], "prepare_draft", FULL_TIME | {"start_date": "30/11/2026",
                                                                  "end_date": "14/12/2026"})
        approvals = {a["role"]: a for a in r.data["draft"]["approvals_required"]}
        self.assertEqual(set(approvals), {LINE_MANAGER, HEAD_OF_PEOPLE})
        for role_id, a in approvals.items():
            self.assertEqual(a["rule_id"], APPROVER_ROLES[role_id].rule_id)
            self.assertEqual(a["basis"], RULES[a["rule_id"]].quote)
        self.assertEqual(approvals[HEAD_OF_PEOPLE]["rule_id"], "long_leave")

    def test_order_of_approvers_does_not_matter(self):
        original = tools.assess_dates

        def reversed_approvers(*args):
            a = original(*args)
            a.approvers.reverse()
            return a
        with mock.patch("agent.tools.assess_dates", reversed_approvers):
            r = run_tool(make_ctx()[0], "prepare_draft", FULL_TIME | {"start_date": "30/11/2026",
                                                                      "end_date": "14/12/2026"})
        first = r.data["draft"]["approvals_required"][0]
        self.assertEqual((first["role"], first["rule_id"]), (HEAD_OF_PEOPLE, "long_leave"))


class InputTests(unittest.TestCase):
    def test_blank_questions_are_refused_before_any_work(self):
        ctx, _ = make_ctx()
        for name, key in (("search_policy", "query"), ("answer_policy", "question")):
            for blank in ("", "   ", "\n\t"):
                with self.subTest(tool=name, value=blank):
                    r = run_tool(ctx, name, {key: blank})
                    self.assertEqual(r.code, Code.INVALID_ARGUMENTS)
        self.assertEqual(ctx.retriever.calls, [])

    def test_dates_in_unrecognised_formats_are_refused(self):
        for start in ("20261123", "2026-W48-1", "11/23/2026", "23/11/26"):
            with self.subTest(start=start):
                r = run_tool(make_ctx()[0], "calculate_dates", {"start_date": start, "end_date": "25/11/2026"})
                self.assertEqual(r.code, Code.INVALID_DATES)

    def test_unverified_answer_text_is_withheld(self):
        ctx, _ = make_ctx(llm=FakeLLM("SECRET-ANSWER 25 days [pay_and_salary_bands#salary-bands]."))
        r = run_tool(ctx, "answer_policy", {"question": "How many annual leave days do permanent UK employees get?"})
        self.assertEqual(r.code, Code.INVALID_CITATIONS)
        self.assertNotIn("SECRET-ANSWER", json.dumps(r.to_dict()))
        self.assertTrue(r.data["answer_withheld"])


if __name__ == "__main__":
    unittest.main()
