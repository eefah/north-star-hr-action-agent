"""Stage 3 graph (agent/graph.py): safety AND effectiveness, offline with fake Pinecone and Groq.
All employees are synthetic. Live checks are in tests/live_evidence.py.

    python -m unittest discover tests
"""
import json
import logging
import shutil
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pinecone.errors import PineconeTimeoutError, ResponseParsingError, UnauthorizedError

from agent import graph as G
from agent.explain import CAN_PROCEED, CANNOT_DETERMINE, CANNOT_PROCEED, NOT_A_DECISION
from agent.rules import HEAD_OF_PEOPLE, LINE_MANAGER, RULES
from agent.tools import ERROR_MESSAGES
from tests.fakes import FakeLLM, make_ctx, make_retriever, without_chunk

FULL_TIME = {"employment_type": "permanent", "uk_based": True, "working_pattern": "full_time"}
LEAVE = FULL_TIME | {"start_date": "23/11/2026", "end_date": "25/11/2026"}
LONG_LEAVE = FULL_TIME | {"start_date": "30/11/2026", "end_date": "14/12/2026"}
SHORT_NOTICE = FULL_TIME | {"start_date": "10/11/2026", "end_date": "12/11/2026"}
EMPLOYEE = "synthetic-employee"
APPROVE = {"decision": "approve", "draft_id": "draft-test-1", "by": EMPLOYEE}
GOOD_ANSWER = "Permanent UK employees get 25 days [annual_leave#eligibility]."
LEAVE_QUESTION = {"question": "How many annual leave days do permanent UK employees get?"}
SECRET = "pcsk_SECRETabcdef123456"


def _raising(exc):
    def fn(*args, **kwargs):
        raise exc
    return fn


def _json_only(value) -> bool:
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_only(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_only(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


class GraphTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.drafts = self.tmp / "drafts"
        self.app = G.build_graph(InMemorySaver())
        self.n = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def gctx(self, **kwargs):
        tool_ctx, _ = make_ctx(**kwargs)
        return G.GraphContext(tool_ctx, self.drafts)

    def start(self, request, kind="leave_request", ctx=None, **kwargs):
        self.n += 1
        self.thread, self.ctx = f"t{self.n}", ctx or self.gctx(**kwargs)
        return G.start(self.app, self.ctx, self.thread, kind, request)

    def resume(self, reply, ctx=None):
        return G.resume(self.app, ctx or self.ctx, self.thread, reply)

    def state(self):
        return self.app.get_state({"configurable": {"thread_id": self.thread}}).values

    def saved_files(self):
        return sorted(self.drafts.glob("*.json")) if self.drafts.exists() else []

    def assertNothingApproved(self):
        s = self.state()
        self.assertNotEqual(s.get("approval", {}).get("status"), "approved")
        self.assertNotEqual(s.get("status"), G.COMPLETED)
        self.assertEqual((self.saved_files(), s.get("actions", [])), ([], []))


# ---------------------------------------------------------------- effectiveness: legitimate requests succeed

class SuccessfulJourneyTests(GraphTestCase):
    def test_end_to_end_leave_request_is_drafted_approved_and_saved(self):
        v = self.start(LEAVE)
        self.assertEqual((v["status"], v["category"], v["reason"]), (G.AWAITING_APPROVAL, CAN_PROCEED, "OK"))
        self.assertEqual(v["pending"]["kind"], "approval")
        self.assertEqual(v["draft"]["dates"]["working_days_requested"], 3)
        self.assertEqual(self.saved_files(), [])  # nothing saved before approval

        v = self.resume(APPROVE)
        self.assertEqual((v["status"], v["category"]), (G.COMPLETED, CAN_PROCEED))
        self.assertIn("NOT been submitted to PeopleHub", v["message"])
        [path] = self.saved_files()
        saved = json.loads(path.read_text())
        self.assertEqual((saved["status"], saved["approved_by"]), (G.SAVED_STATUS, EMPLOYEE))
        s = self.state()
        self.assertEqual(s["actions"], [{"action": "save_draft", "draft_id": "draft-test-1", "result": "saved"}])
        self.assertEqual(s["decisions"], {"eligibility": "OK", "dates": "OK", "draft": "OK"})
        self.assertEqual({r["chunk_id"] for r in s["evidence_refs"]},
                         {"annual_leave#eligibility", "annual_leave#contact", "annual_leave#how-to-request"})
        self.assertTrue(all(r["doc_id"] == "NS-ANNUAL-LEAVE" and r["version"] == "2.0" for r in s["evidence_refs"]))

    def test_long_leave_needs_head_of_people_and_employee_approval(self):
        v = self.start(LONG_LEAVE)
        roles = {a["role"]: a for a in v["draft"]["approvals_required"]}
        self.assertEqual(set(roles), {LINE_MANAGER, HEAD_OF_PEOPLE})
        self.assertEqual(roles[HEAD_OF_PEOPLE]["basis"], RULES["long_leave"].quote)
        self.assertEqual(v["pending"]["kind"], "approval")
        self.assertEqual(self.resume(APPROVE)["status"], G.COMPLETED)

    def test_missing_details_are_asked_for_then_the_request_succeeds(self):
        v = self.start({"employment_type": "permanent", "start_date": "23/11/2026", "end_date": "25/11/2026"})
        self.assertEqual((v["status"], v["pending"]["missing"]), (G.AWAITING_INFO, ["uk_based", "working_pattern"]))
        v = self.resume({"uk_based": True, "working_pattern": "part_time"})
        self.assertEqual(v["status"], G.AWAITING_APPROVAL)
        self.assertTrue(any("pro-rated" in n for n in v["draft"]["eligibility_notes"]))

    def test_policy_question_is_answered_with_citations(self):
        v = self.start(LEAVE_QUESTION, kind="policy_question", llm=FakeLLM(GOOD_ANSWER))
        self.assertEqual((v["status"], v["category"], v["citations"]), (G.ANSWERED, CAN_PROCEED,
                                                                        ["annual_leave#eligibility"]))
        self.assertIn("25 days", v["message"])


# ---------------------------------------------------------------- AC4: three distinct, explained outcomes

class ExplainabilityTests(GraphTestCase):
    def test_cannot_proceed_not_eligible_is_explained_by_the_rule(self):
        v = self.start(LEAVE | {"employment_type": "contractor"})
        self.assertEqual((v["status"], v["category"], v["reason"]), (G.REFUSED, CANNOT_PROCEED, "NOT_ELIGIBLE"))
        self.assertIn(RULES["contractors"].quote, v["message"])
        self.assertIn("annual_leave#eligibility", v["message"])
        self.assertNotIn("dates", self.state()["decisions"])  # stopped at the failed check
        self.assertNothingApproved()

    def test_cannot_proceed_notice_too_short_explains_notice_and_offers_a_date(self):
        v = self.start(SHORT_NOTICE)
        self.assertEqual((v["category"], v["reason"]), (CANNOT_PROCEED, "NOTICE_TOO_SHORT"))
        self.assertIn(RULES["notice"].quote, v["message"])
        self.assertIn("2026-11-17", v["message"])
        self.assertIn("2026-11-17", v["recovery"])
        self.assertNothingApproved()

    def test_cannot_proceed_invalid_dates(self):
        v = self.start(LEAVE | {"start_date": "25/11/2026", "end_date": "23/11/2026"})
        self.assertEqual((v["category"], v["reason"]), (CANNOT_PROCEED, "INVALID_DATES"))
        self.assertIn("end date is before the start date", v["message"])

    def test_cannot_determine_is_never_described_as_ineligible(self):
        cases = [(LEAVE | {"employment_type": "other"}, {}, "UNSUPPORTED_EMPLOYMENT_TYPE"),
                 (LEAVE | {"uk_based": False}, {}, "UNSUPPORTED_LOCATION"),
                 (LEAVE, {"retriever": make_retriever(without_chunk("annual_leave#eligibility"))},
                  "INSUFFICIENT_EVIDENCE")]
        for request, kwargs, reason in cases:
            with self.subTest(reason=reason):
                v = self.start(request, **kwargs)
                self.assertEqual((v["status"], v["category"], v["reason"]), (G.UNDETERMINED, CANNOT_DETERMINE,
                                                                             reason))
                self.assertIn(NOT_A_DECISION, v["message"])
                for wording in ("can't go ahead", "not eligible", "ineligible", "outside this policy"):
                    self.assertNotIn(wording, v["message"].lower())
                self.assertIn("people@northstar-demo.example", v["recovery"])
                self.assertNothingApproved()

    def test_cannot_determine_technical_failure_is_a_technical_message(self):
        v = self.start(LEAVE, retriever=make_retriever(fail_times=99))
        self.assertEqual((v["status"], v["pending"]["kind"]), (G.AWAITING_RETRY, "retry"))
        self.assertEqual(v["pending"]["message"], ERROR_MESSAGES["transient"])
        v = self.resume({"retry": False})
        self.assertEqual((v["status"], v["category"], v["reason"]), (G.UNDETERMINED, CANNOT_DETERMINE,
                                                                     "TOOL_FAILURE"))
        self.assertEqual(v["message"], ERROR_MESSAGES["transient"])
        self.assertEqual(v["recovery"], "Try again in a few minutes.")
        self.assertNothingApproved()

    def test_no_evidence_question_is_an_explicit_refusal_to_answer(self):
        llm = FakeLLM("should never be used")
        v = self.start({"question": "zzqx qqzx"}, kind="policy_question", llm=llm)
        self.assertEqual((v["category"], v["reason"], llm.calls), (CANNOT_DETERMINE, "NO_EVIDENCE", []))

    def test_every_tool_code_has_an_explanation(self):
        from agent.explain import CATEGORY
        from agent.tools import Code
        self.assertEqual(set(CATEGORY), set(Code))


# ---------------------------------------------------------------- AC3 (offline part): citations

class CitationTests(GraphTestCase):
    def test_invalid_citations_are_never_shown_or_saved(self):
        llm = FakeLLM("SECRET-ANSWER 25 days [pay_and_salary_bands#salary-bands].")
        v = self.start(LEAVE_QUESTION, kind="policy_question", llm=llm)
        self.assertEqual((v["category"], v["reason"]), (CANNOT_DETERMINE, "INVALID_CITATIONS"))
        self.assertNotIn("SECRET-ANSWER", json.dumps(v))
        self.assertNotIn("SECRET-ANSWER", json.dumps(self.state()))

    def test_evidence_references_are_retrieved_verified_chunks(self):
        self.start(LEAVE)
        refs = self.state()["evidence_refs"]
        cited = {c for e in self.state()["trace"] for c in e.get("citations", [])}
        self.assertEqual({r["chunk_id"] for r in refs}, cited)


# ---------------------------------------------------------------- AC2: controlled failure

class ControlledFailureTests(GraphTestCase):
    def broken_retriever(self, exc):
        r = make_retriever()
        r.search = _raising(exc)
        return r

    def check_controlled(self, v):
        text = json.dumps(v)
        for leak in (SECRET, "Traceback", "Error:", "pcsk_"):
            self.assertNotIn(leak, text)
        self.assertNothingApproved()

    def test_authentication_failure_stops_without_retry_offer(self):
        with self.assertLogs("northstar.diagnostics", logging.ERROR) as logs:
            v = self.start(LEAVE, retriever=self.broken_retriever(UnauthorizedError(f"bad key {SECRET}")))
        self.assertEqual((v["status"], v["reason"], v["pending"]), (G.UNDETERMINED, "TOOL_FAILURE", None))
        self.assertEqual(self.state()["trace"][1]["attempts"], 1)
        self.assertNotIn(SECRET, "\n".join(logs.output))
        self.check_controlled(v)

    def test_timeouts_and_outages_offer_a_bounded_retry(self):
        for exc in (PineconeTimeoutError("timed out"), ConnectionError("down")):
            with self.subTest(e=type(exc).__name__):
                v = self.start(LEAVE, retriever=self.broken_retriever(exc))
                for _ in range(self.ctx.max_retry_rounds):
                    self.assertEqual(v["status"], G.AWAITING_RETRY)
                    v = self.resume({"retry": True})
                self.assertEqual((v["status"], v["reason"]), (G.UNDETERMINED, "TOOL_FAILURE"))  # retries exhausted
                self.assertEqual(self.state()["retry_rounds"], self.ctx.max_retry_rounds)
                self.check_controlled(v)

    def test_recovery_after_a_temporary_outage(self):
        v = self.start(LEAVE, retriever=make_retriever(fail_times=3))  # the first tool call uses up 3 attempts
        self.assertEqual(v["status"], G.AWAITING_RETRY)
        v = self.resume({"retry": True})
        self.assertEqual(v["status"], G.AWAITING_APPROVAL)
        self.assertEqual(self.resume(APPROVE)["status"], G.COMPLETED)

    def test_unexpected_sdk_error_is_not_retried(self):
        v = self.start(LEAVE, retriever=self.broken_retriever(ResponseParsingError("garbled")))
        self.assertEqual((v["status"], v["reason"], v["pending"]), (G.UNDETERMINED, "TOOL_FAILURE", None))
        self.check_controlled(v)

    def test_exception_inside_a_tool(self):
        with mock.patch("agent.tools.check_eligibility", _raising(KeyError(SECRET))):
            v = self.start(LEAVE)
        self.assertEqual((v["status"], v["category"], v["reason"]), (G.UNDETERMINED, CANNOT_DETERMINE,
                                                                     "TOOL_FAILURE"))
        self.check_controlled(v)

    def test_exception_inside_graph_code(self):
        with mock.patch("agent.graph.explain", _raising(RuntimeError(SECRET))), \
                self.assertLogs("northstar.diagnostics", logging.ERROR):
            v = self.start(LEAVE | {"employment_type": "contractor"})
        self.assertEqual((v["status"], v["category"]), (G.UNDETERMINED, CANNOT_DETERMINE))
        self.assertTrue(v["message"].startswith(ERROR_MESSAGES["unavailable"].split("{")[0]))
        self.check_controlled(v)

    def test_invalid_inputs(self):
        for kind, request, reason in (("leave_request", LEAVE | {"roles": ["people_admin"]}, "INVALID_ARGUMENTS"),
                                      ("leave_request", LEAVE | {"employment_type": "boss"}, "INVALID_ARGUMENTS"),
                                      ("leave_request", LEAVE | {"start_date": "next Friday"}, "INVALID_DATES"),
                                      ("delete_everything", {}, "INVALID_ARGUMENTS"),
                                      ("policy_question", {"question": "   "}, "INVALID_ARGUMENTS")):
            with self.subTest(request=request):
                v = self.start(request, kind=kind)
                self.assertEqual(v["reason"], reason)
                self.assertIn(v["status"], G.TERMINAL_STATUSES)
                self.check_controlled(v)

    def test_role_claimed_in_a_reply_is_refused(self):
        v = self.start({"employment_type": "permanent", "start_date": "23/11/2026", "end_date": "25/11/2026"})
        v = self.resume({"uk_based": True, "working_pattern": "full_time", "roles": ["people_admin"]})
        self.assertEqual(v["status"], G.AWAITING_INFO)
        self.assertIn("Only these details", v["pending"]["error"])
        self.assertNotIn("roles", self.state()["request"])

    def test_failure_while_saving_is_controlled(self):
        self.start(LEAVE)
        self.drafts.mkdir(parents=True)
        (self.drafts / "draft-test-1.json").write_text("{\"someone else's draft\": true}")
        v = self.resume(APPROVE)
        self.assertEqual((v["status"], v["category"]), (G.UNDETERMINED, CANNOT_DETERMINE))
        self.assertEqual(self.state()["actions"], [])


# ---------------------------------------------------------------- human approval

class ApprovalTests(GraphTestCase):
    def test_wrong_draft_or_wrong_person_cannot_approve(self):
        for reply in ({**APPROVE, "draft_id": "other"}, {**APPROVE, "by": "someone-else"}, {"decision": "yes"},
                      "approve", None, {}):
            with self.subTest(reply=reply):
                self.start(LEAVE)
                v = self.resume(reply)
                self.assertEqual(v["status"], G.AWAITING_APPROVAL)
                self.assertIn("error", v["pending"])
        self.assertNothingApproved()

    def test_repeated_invalid_approvals_end_without_saving(self):
        self.start(LEAVE)
        for _ in range(self.ctx.max_invalid_replies):
            v = self.resume({"decision": "approve", "draft_id": "nope", "by": EMPLOYEE})
        self.assertEqual(v["status"], G.UNDETERMINED)
        self.assertNothingApproved()

    def test_rejecting_the_draft_saves_nothing(self):
        self.start(LEAVE)
        v = self.resume({**APPROVE, "decision": "reject"})
        self.assertEqual((v["status"], v["category"]), (G.CANCELLED, "cancelled"))
        self.assertNothingApproved()

    def test_approval_on_a_later_day_rechecks_the_notice_rule(self):
        self.start(LEAVE)  # prepared on 3 Nov for 23 Nov: enough notice then
        later = self.gctx(today=date(2026, 11, 12))  # by 12 Nov, 23 Nov no longer gives 10 working days
        v = self.resume(APPROVE, ctx=later)
        self.assertEqual((v["status"], v["reason"]), (G.REFUSED, "NOTICE_TOO_SHORT"))
        self.assertEqual(self.state()["approval"]["status"], "expired")
        self.assertNothingApproved()


# ---------------------------------------------------------------- AC1: recoverability

class RecoverabilityTests(GraphTestCase):
    def test_every_checkpoint_holds_only_plain_data(self):
        self.start(LONG_LEAVE)
        self.resume(APPROVE)
        history = list(self.app.get_state_history({"configurable": {"thread_id": self.thread}}))
        self.assertGreater(len(history), 5)
        for snap in history:
            self.assertTrue(_json_only(snap.values), snap.next)
            json.dumps(snap.values)

    def test_resume_in_a_new_process_with_sqlite(self):
        db = self.tmp / "checkpoints.sqlite"
        with sqlite3.connect(db, check_same_thread=False) as conn:
            app = G.build_graph(SqliteSaver(conn))
            first = self.gctx()
            v = G.start(app, first, "resume-1", "leave_request", LONG_LEAVE)
            self.assertEqual(v["status"], G.AWAITING_APPROVAL)
            before = app.get_state({"configurable": {"thread_id": "resume-1"}}).values

        # Simulated restart: new connection, new graph, new clients with an empty evidence store.
        with sqlite3.connect(db, check_same_thread=False) as conn:
            blobs = b"".join(row[0] for row in conn.execute("select checkpoint from checkpoints")) + \
                b"".join(row[0] or b"" for row in conn.execute("select value from writes"))
            for forbidden in (b"ToolContext", b"HybridRetriever", b"FakeLLM", b"FakeIndex", b"lambda",
                              b"GraphContext", b"EvidenceStore"):
                self.assertNotIn(forbidden, blobs)

            app = G.build_graph(SqliteSaver(conn))
            fresh = self.gctx()
            self.assertEqual(fresh.tools.evidence.chunks, {})
            after = app.get_state({"configurable": {"thread_id": "resume-1"}}).values
            for key in ("evidence_refs", "decisions", "approval", "draft", "request", "trace"):
                self.assertEqual(after[key], before[key], key)
            self.assertEqual(after["approval"]["status"], "pending")

            v = G.resume(app, fresh, "resume-1", APPROVE)
            self.assertEqual(v["status"], G.COMPLETED)
            final = app.get_state({"configurable": {"thread_id": "resume-1"}}).values
            self.assertEqual(final["approval"]["status"], "approved")
            self.assertEqual(len(self.saved_files()), 1)

            # Resuming again, or replaying the approval, does not repeat the action.
            v = G.resume(app, fresh, "resume-1", APPROVE)
            again = app.get_state({"configurable": {"thread_id": "resume-1"}}).values
            self.assertEqual((v["status"], len(again["actions"]), len(self.saved_files())), (G.COMPLETED, 1, 1))

    def test_save_is_idempotent_if_the_node_runs_twice(self):
        self.start(LEAVE)
        self.resume(APPROVE)
        state, ctx = self.state(), self.ctx
        rt = mock.Mock(context=ctx)
        second = G.save_draft(state, rt)  # e.g. a crash after writing, before the checkpoint was saved
        self.assertEqual(second["actions"][0]["result"], "already_saved")
        self.assertEqual(len(self.saved_files()), 1)

    def test_paused_for_details_survives_a_restart(self):
        db = self.tmp / "cp.sqlite"
        with sqlite3.connect(db, check_same_thread=False) as conn:
            G.start(G.build_graph(SqliteSaver(conn)), self.gctx(), "info-1", "leave_request",
                    {"employment_type": "permanent", "start_date": "23/11/2026", "end_date": "25/11/2026"})
        with sqlite3.connect(db, check_same_thread=False) as conn:
            v = G.resume(G.build_graph(SqliteSaver(conn)), self.gctx(), "info-1",
                         {"uk_based": True, "working_pattern": "full_time"})
        self.assertEqual(v["status"], G.AWAITING_APPROVAL)

    def test_threads_cannot_be_restarted_or_invented(self):
        self.start(LEAVE)
        with self.assertRaises(ValueError):
            G.start(self.app, self.ctx, self.thread, "leave_request", LEAVE)
        with self.assertRaises(ValueError):
            G.resume(self.app, self.ctx, "no-such-thread", APPROVE)


class EntryPointTests(GraphTestCase):
    def test_an_error_raised_by_langgraph_itself_is_controlled(self):
        self.start(LEAVE)
        with mock.patch.object(type(self.app), "invoke", _raising(RuntimeError(SECRET))), \
                self.assertLogs("northstar.diagnostics", logging.ERROR):
            v = self.resume(APPROVE)
        self.assertEqual((v["reason"], v["category"]), ("TOOL_FAILURE", CANNOT_DETERMINE))
        self.assertNotIn(SECRET, json.dumps(v))
        self.assertEqual(v["status"], G.AWAITING_APPROVAL)  # the paused workflow is intact
        self.assertNothingApproved()
        self.assertEqual(self.resume(APPROVE)["status"], G.COMPLETED)  # and can still be completed


class EmployeeViewTests(GraphTestCase):
    def test_view_never_contains_trace_or_diagnostics(self):
        v = self.start(LEAVE, retriever=make_retriever(fail_times=99))
        self.assertEqual(set(v), {"status", "category", "reason", "message", "recovery", "citations", "pending",
                                  "draft"})
        self.assertNotIn("errors", json.dumps(v))
        self.assertNotIn("diagnostic ref", json.dumps(v))


if __name__ == "__main__":
    unittest.main()
