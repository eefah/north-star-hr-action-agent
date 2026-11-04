"""LIVE checks against the real Pinecone index and Groq. Not part of the offline suite (the file name doesn't
match test*.py, so `discover` skips it). Costs a few Pinecone queries and two small Groq calls. Run with

    python -m unittest -v tests.live_evidence

Known questions and known policy passages: each fixed evidence query must actually retrieve the chunk its
rules depend on, as an employee, today, and that chunk must still pass verify_evidence.
"""
import os
import unittest
from datetime import date

import config
from agent.dates import BankHolidays
from agent.rules import APPROVER_ROLES, DATE_RULES, ELIGIBILITY_RULES, HEAD_OF_PEOPLE, LINE_MANAGER, RULES, \
    verify_evidence
from agent.tools import EVIDENCE_QUERIES, Code, ToolContext, run_tool

LIVE = bool(os.environ.get("PINECONE_API_KEY") and os.environ.get("GROQ_API_KEY"))  # config loads .env
FULL_TIME = {"employment_type": "permanent", "uk_based": True, "working_pattern": "full_time"}
ROLES = ("employee",)

KNOWN_QUESTIONS = [  # (question, chunk the answer must cite, text the answer must contain)
    ("How many days of paid annual leave do permanent UK employees get?", "annual_leave#eligibility", "25"),
    ("How much notice do I need to give when requesting annual leave?", "annual_leave#how-to-request", "10"),
]


@unittest.skipUnless(LIVE, "PINECONE_API_KEY and GROQ_API_KEY are needed for live checks")
class LiveEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rag import RAG
        cls.rag = RAG()
        cls.retriever = cls.rag.retriever
        cls.today = date.today()

    def ctx(self):
        return ToolContext(roles=ROLES, today=self.today, retriever=self.retriever, rag=self.rag,
                           holidays=BankHolidays.load())

    def test_each_fixed_query_retrieves_its_required_chunk(self):
        for chunk_id, query in EVIDENCE_QUERIES.items():
            with self.subTest(chunk=chunk_id):
                result = self.retriever.search(query, list(ROLES), self.today)
                ids = [h.id for h in result.candidates]
                print(f"\n  {chunk_id}: outcome={result.outcome} candidate rank="
                      f"{ids.index(chunk_id) + 1 if chunk_id in ids else 'MISSING'} of {len(ids)}; "
                      f"in final hits={chunk_id in [h.id for h in result.hits]}")
                self.assertIn(chunk_id, ids)
                chunk = next(h.meta for h in result.candidates if h.id == chunk_id)
                rules = [r for r in ELIGIBILITY_RULES + DATE_RULES if RULES[r].chunk_id == chunk_id]
                check = verify_evidence({chunk_id: chunk}, rules)
                self.assertTrue(check.ok, check.problems)

    def test_retrieved_candidates_are_all_authorised(self):
        flt = self.retriever.build_filter(list(ROLES), self.today)
        result = self.retriever.search("Band 3 salary range disciplinary appeal", list(ROLES), self.today)
        self.assertTrue(all(self.retriever.matches(h.meta, flt) for h in result.candidates))
        self.assertFalse(any(h.id.startswith(("pay_and_salary_bands", "disciplinary_procedure"))
                             for h in result.candidates))

    def test_tools_end_to_end(self):
        ctx = self.ctx()
        elig = run_tool(ctx, "check_eligibility", FULL_TIME)
        self.assertEqual(elig.code, Code.OK, elig.message)
        self.assertEqual(set(elig.citations), {"annual_leave#eligibility", "annual_leave#contact"})

        draft = run_tool(ctx, "prepare_draft", FULL_TIME | {"start_date": "30/11/2026", "end_date": "14/12/2026"})
        self.assertEqual(draft.code, Code.OK, draft.message)
        approvals = {a["role"]: a for a in draft.data["draft"]["approvals_required"]}
        self.assertEqual(set(approvals), {LINE_MANAGER, HEAD_OF_PEOPLE})
        self.assertEqual(approvals[HEAD_OF_PEOPLE]["rule_id"], APPROVER_ROLES[HEAD_OF_PEOPLE].rule_id)
        print(f"\n  draft: {draft.data['draft']['dates']['working_days_requested']} working days, "
              f"external calls={elig.attempts + draft.attempts}, retries={len(elig.errors + draft.errors)}")

        short = run_tool(ctx, "prepare_draft", FULL_TIME | {"start_date": "09/11/2026", "end_date": "10/11/2026"})
        self.assertEqual((short.code, short.reason), (Code.PRECONDITION_FAILED, Code.NOTICE_TOO_SHORT))

    def test_known_questions_cite_the_known_passages(self):
        ctx = self.ctx()
        for question, chunk_id, must_contain in KNOWN_QUESTIONS:
            with self.subTest(question=question):
                r = run_tool(ctx, "answer_policy", {"question": question})
                print(f"\n  Q: {question}\n  code={r.code.value} citations={r.citations}\n  A: {r.message[:300]}")
                self.assertEqual(r.code, Code.OK)
                self.assertIn(chunk_id, r.citations)
                self.assertIn(must_contain, r.message)


@unittest.skipUnless(LIVE, "PINECONE_API_KEY and GROQ_API_KEY are needed for live checks")
class LiveGraphTests(unittest.TestCase):
    """Stage 3 graph against the real Pinecone index, reranker and Groq. Drafts go to a temporary folder."""
    @classmethod
    def setUpClass(cls):
        import shutil
        import tempfile
        from pathlib import Path
        from langgraph.checkpoint.memory import InMemorySaver
        from agent import graph
        from rag import RAG
        cls.G, cls.rag, cls.today = graph, RAG(), date.today()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.app = graph.build_graph(InMemorySaver())
        cls.n = 0

    def gctx(self, rag=None, retriever=None, **kwargs):
        rag = rag or self.rag
        tools = ToolContext(roles=ROLES, today=self.today, retriever=retriever or rag.retriever, rag=rag,
                            holidays=BankHolidays.load(), **kwargs)
        return self.G.GraphContext(tools, self.tmp / "drafts")

    def run_request(self, kind, request, ctx=None):
        type(self).n += 1
        self.thread = f"live-{self.n}"
        return self.G.start(self.app, ctx or self.gctx(), self.thread, kind, request)

    def state(self):
        return self.app.get_state({"configurable": {"thread_id": self.thread}}).values

    def test_legitimate_request_completes_with_verified_evidence(self):
        ctx = self.gctx()
        v = self.run_request("leave_request", FULL_TIME | {"start_date": "30/11/2026", "end_date": "14/12/2026"},
                             ctx)
        self.assertEqual((v["status"], v["category"]), (self.G.AWAITING_APPROVAL, "can_proceed"), v["message"])
        refs = self.state()["evidence_refs"]
        print(f"\n  evidence: {[(r['chunk_id'], r['doc_id'], r['version'], r['status']) for r in refs]}")
        self.assertEqual({r["chunk_id"] for r in refs},
                         {"annual_leave#eligibility", "annual_leave#contact", "annual_leave#how-to-request"})
        self.assertTrue(all((r["doc_id"], r["version"], r["status"]) == ("NS-ANNUAL-LEAVE", "2.0", "current")
                            for r in refs))
        draft = v["draft"]
        self.assertEqual({a["role"] for a in draft["approvals_required"]}, {LINE_MANAGER, HEAD_OF_PEOPLE})
        v = self.G.resume(self.app, ctx, self.thread,
                          {"decision": "approve", "draft_id": draft["draft_id"], "by": ctx.tools.employee_ref})
        self.assertEqual(v["status"], self.G.COMPLETED)
        self.assertEqual(len(self.state()["actions"]), 1)

    def test_known_rule_refusal(self):
        v = self.run_request("leave_request", FULL_TIME | {"employment_type": "contractor",
                                                           "start_date": "30/11/2026", "end_date": "02/12/2026"})
        self.assertEqual((v["category"], v["reason"]), ("cannot_proceed", "NOT_ELIGIBLE"))
        self.assertIn(RULES["contractors"].quote, v["message"])

    def test_known_question_is_answered_with_a_supporting_citation(self):
        v = self.run_request("policy_question", {"question": KNOWN_QUESTIONS[0][0]})
        print(f"\n  answer: {v['message'][:200]}\n  citations: {v['citations']}")
        self.assertEqual((v["status"], v["citations"]), (self.G.ANSWERED, ["annual_leave#eligibility"]))
        self.assertIn("25", v["message"])
        chunk = self.rag.retriever.by_id["annual_leave#eligibility"]
        self.assertIn("25 days of paid annual leave", chunk["chunk_text"])  # the cited passage supports the claim

    def test_question_outside_the_policies_is_refused_not_guessed(self):
        v = self.run_request("policy_question", {"question": "What is the company policy on bringing pets "
                                                             "to the office?"})
        print(f"\n  out-of-scope: reason={v['reason']} message={v['message'][:120]}")
        self.assertEqual((v["category"], v["reason"]), ("cannot_determine", "NO_EVIDENCE"))

    def test_restricted_policy_is_not_used_for_an_employee(self):
        v = self.run_request("policy_question", {"question": "What is the salary range for Band 3?"})
        print(f"\n  restricted: reason={v['reason']} citations={v['citations']}")
        self.assertFalse(any(c.startswith("pay_and_salary_bands") for c in v["citations"]))
        if not v["citations"]:  # nothing citable for this employee: must be a refusal, not an uncited answer
            self.assertEqual((v["category"], v["reason"]), ("cannot_determine", "NO_EVIDENCE"))

    def test_no_policy_in_force_means_insufficient_evidence(self):
        v = self.run_request("leave_request", FULL_TIME | {"start_date": "30/11/2026", "end_date": "02/12/2026"},
                             self.gctx(as_of=date(2026, 9, 15)))  # before annual leave v2.0 came into force
        self.assertEqual((v["category"], v["reason"]), ("cannot_determine", "INSUFFICIENT_EVIDENCE"))
        self.assertNotIn("not eligible", v["message"].lower())

    def test_retrieval_failure_is_distinguished_from_absent_evidence(self):
        """A real Pinecone call with a deliberately invalid key: a genuine authentication failure."""
        import copy
        from pinecone import Pinecone
        broken = copy.copy(self.rag.retriever)
        broken.index = Pinecone(api_key="pcsk_deliberately_invalid_key_for_test").Index(
            host=self.rag.retriever.index._host)
        v = self.run_request("leave_request", FULL_TIME | {"start_date": "30/11/2026", "end_date": "02/12/2026"},
                             self.gctx(retriever=broken))
        trace = self.state()["trace"]
        print(f"\n  broken retrieval: reason={v['reason']} failure={trace[1].get('failure')} "
              f"attempts={trace[1].get('attempts')}")
        self.assertEqual((v["category"], v["reason"]), ("cannot_determine", "TOOL_FAILURE"))
        self.assertEqual((trace[1]["failure"], trace[1]["attempts"]), ("rejected", 1))  # not retried
        self.assertNotIn("deliberately_invalid", str(v) + str(trace))

    def test_invalid_citation_answer_is_withheld(self):
        """Live retrieval; the LLM reply is scripted, because a real model can't be made to mis-cite on demand."""
        from tests.fakes import FakeLLM, make_rag
        rag = make_rag(self.rag.retriever, FakeLLM("WITHHELD-TEXT 25 days [pay_and_salary_bands#salary-bands]."))
        v = self.run_request("policy_question", {"question": KNOWN_QUESTIONS[0][0]}, self.gctx(rag=rag))
        self.assertEqual((v["category"], v["reason"]), ("cannot_determine", "INVALID_CITATIONS"))
        self.assertNotIn("WITHHELD-TEXT", str(v) + str(self.state()))


if __name__ == "__main__":
    unittest.main()
