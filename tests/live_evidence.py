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


if __name__ == "__main__":
    unittest.main()
