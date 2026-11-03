"""Offline regression tests: no Pinecone or Groq calls. Run with

    python -m unittest discover tests
"""
import unittest
from datetime import date
from types import SimpleNamespace

import config
from rag import INVISIBLE, NO_EVIDENCE_ANSWER, RAG, parse_citations
from retriever import Hit, HybridRetriever, RetrievalResult, select

TODAY = date(2026, 10, 18)


def hit(cid, rerank, rrf=0.03, dense=None, bm25=None):
    return Hit(cid, "text", {}, dense_rank=dense, bm25_rank=bm25, rrf_score=rrf, rerank_score=rerank)


def chunk(cid, text, roles=("employee", "people_admin"), tenant=config.NAMESPACE, status="current",
          effective_from=20261001, effective_until=99991231):
    return {"_id": cid, config.TEXT_FIELD: text, "acl_roles": list(roles), "tenant_id": tenant,
            "status": status, "effective_from": effective_from, "effective_until": effective_until}


class FakeIndex:
    """Stands in for Pinecone search: returns the given IDs in order, ignoring the filter."""
    def __init__(self, ids):
        self.ids = ids

    def search(self, **kwargs):
        return SimpleNamespace(result=SimpleNamespace(hits=[SimpleNamespace(id=i) for i in self.ids]))


class FakeInference:
    """Stands in for the Pinecone reranker: scores each document by looking up its text."""
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, documents, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=i, score=self.scores.get(d, 0.0))
                                     for i, d in enumerate(documents)])


def fake_retriever(chunks, dense_ids, scores):
    r = HybridRetriever.__new__(HybridRetriever)  # skip __init__: no Pinecone client, no chunks file
    r.chunks, r.by_id = chunks, {c["_id"]: c for c in chunks}
    r.index, r.inference = FakeIndex(dense_ids), FakeInference(scores)
    return r


class SelectTests(unittest.TestCase):
    def q01_candidates(self):
        """Scores from the Q01 trace. onboarding#first-month has higher RRF than two chunks that were sent."""
        return [hit("annual_leave#eligibility", 0.997, 0.0328), hit("sickness_absence#company-sick-pay", 0.265, 0.0318),
                hit("parental_leave#scope-and-eligibility", 0.208, 0.0306), hit("onboarding#first-month", 0.018, 0.0303),
                hit("annual_leave#carry-over", 0.026, 0.0298), hit("annual_leave#how-to-request", 0.022, 0.0296),
                hit("remote_work#security", 0.004, 0.0290)]

    def test_selects_by_rerank_score_not_rrf(self):
        final, outcome = select(self.q01_candidates(), 5, 0.01, 0)
        self.assertEqual([h.id for h in final], ["annual_leave#eligibility", "sickness_absence#company-sick-pay",
                                                 "parental_leave#scope-and-eligibility", "annual_leave#carry-over",
                                                 "annual_leave#how-to-request"])
        self.assertEqual(outcome, "evidence")

    def test_reasons_distinguish_threshold_from_cap(self):
        cands = self.q01_candidates()
        select(cands, 5, 0.01, 0)
        by_id = {h.id: h for h in cands}
        self.assertTrue(by_id["annual_leave#eligibility"].reason.startswith("sent: 1 of max 5"))
        capped = by_id["onboarding#first-month"]
        self.assertTrue(capped.passed_threshold)
        self.assertFalse(capped.sent)
        self.assertIn("beyond the 5-chunk cap", capped.reason)
        low = by_id["remote_work#security"]
        self.assertFalse(low.passed_threshold)
        self.assertIn("below 0.01", low.reason)

    def test_cap_is_a_maximum_not_a_target(self):
        final, _ = select([hit("a#x", 0.9), hit("b#y", 0.005), hit("c#z", 0.001)], 5, 0.01, 0)
        self.assertEqual([h.id for h in final], ["a#x"])

    def test_nothing_above_threshold(self):
        final, outcome = select([hit("a#x", 0.004, dense=1, bm25=3)], 5, 0.01, 0)
        self.assertEqual((final, outcome), ([], "below_threshold"))

    def test_first_stage_agreement_flags_possible_retrieval_failure(self):
        final, outcome = select([hit("a#x", 0.003, dense=1, bm25=1)], 5, 0.01, 0)
        self.assertEqual((final, outcome), ([], "check_retrieval"))

    def test_rescue_needs_both_first_stage_ranks(self):
        cands = [hit("a#x", 0.5), hit("both#top", 0.003, dense=1, bm25=1), hit("dense#only", 0.004, dense=1)]
        self.assertEqual([h.id for h in select(cands, 5, 0.01, 0)[0]], ["a#x"])
        final, _ = select(cands, 5, 0.01, 1)
        self.assertEqual([h.id for h in final], ["a#x", "both#top"])
        self.assertIn("rescued", final[1].reason)

    def test_rescue_respects_cap(self):
        cands = [hit(f"d#{i}", 0.9 - i / 10) for i in range(5)] + [hit("both#top", 0.003, dense=1, bm25=1)]
        final, _ = select(cands, 5, 0.01, 1)
        self.assertNotIn("both#top", [h.id for h in final])

    def test_reselect_resets_flags(self):
        cands = [hit("a#x", 0.5), hit("b#y", 0.05)]
        select(cands, 5, 0.0, 0)
        select(cands, 5, 0.1, 0)
        self.assertEqual([h.sent for h in cands], [True, False])


class AccessControlTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("annual_leave#eligibility", "annual leave days entitlement"),
            chunk("pay_and_salary_bands#salary-bands", "band 3 salary range", roles=("people_admin",)),
            chunk("other_tenant#leave", "annual leave days", tenant="someone_else"),
            chunk("expenses_expired#deadline", "expenses deadline 60 days", status="expired", effective_until=20260930),
            chunk("future#leave", "annual leave days from next year", effective_from=20270101),
        ]

    def flt(self, roles, include_expired=False):
        return HybridRetriever.build_filter(roles, TODAY, include_expired)

    def test_matches_enforces_role_tenant_status_and_dates(self):
        allowed = [c["_id"] for c in self.chunks if HybridRetriever.matches(c, self.flt(["employee"]))]
        self.assertEqual(allowed, ["annual_leave#eligibility"])
        admin = [c["_id"] for c in self.chunks if HybridRetriever.matches(c, self.flt(["employee", "people_admin"]))]
        self.assertEqual(admin, ["annual_leave#eligibility", "pay_and_salary_bands#salary-bands"])

    def test_bm25_never_returns_restricted_chunks(self):
        r = fake_retriever(self.chunks, [], {})
        self.assertNotIn("pay_and_salary_bands#salary-bands", r.bm25("band 3 salary range", self.flt(["employee"]), 20))

    def test_search_drops_chunks_the_remote_filter_should_have_hidden(self):
        """Even if Pinecone returned a restricted or expired chunk, it must not reach the reranker or the LLM."""
        dense = ["pay_and_salary_bands#salary-bands", "expenses_expired#deadline", "other_tenant#leave",
                 "annual_leave#eligibility"]
        scores = {c[config.TEXT_FIELD]: 0.9 for c in self.chunks}
        res = fake_retriever(self.chunks, dense, scores).search("band 3 salary range annual leave", ["employee"], TODAY)
        self.assertEqual([h.id for h in res.candidates], ["annual_leave#eligibility"])
        self.assertEqual([h.id for h in res.hits], ["annual_leave#eligibility"])

    def test_access_denied_question_gets_no_candidates(self):
        res = fake_retriever(self.chunks, ["pay_and_salary_bands#salary-bands"], {}).search(
            "band 3 salary range", ["employee"], TODAY)
        self.assertEqual((res.hits, res.outcome), ([], "no_candidates"))


class RefusalGateTests(unittest.TestCase):
    def rag_with(self, outcome):
        rag = RAG.__new__(RAG)
        rag.retriever = SimpleNamespace(search=lambda *a, **k: RetrievalResult("q", {}, [], [], outcome=outcome))
        rag.llm = None  # any LLM call would raise AttributeError
        return rag

    def test_no_evidence_means_fixed_refusal_and_no_llm_call(self):
        for outcome in ("no_candidates", "below_threshold", "check_retrieval"):
            a = self.rag_with(outcome).answer("What is the salary range for Band 3?", ["employee"], TODAY)
            self.assertEqual(a.text, NO_EVIDENCE_ANSWER)  # same words whatever the reason: nothing is disclosed
            self.assertEqual((a.hits, a.model, a.retrieval.outcome), ([], None, outcome))


class CitationTests(unittest.TestCase):
    allowed = {"parental_leave#how-to-request", "parental_leave#partner-benefit"}

    def test_zero_width_space_and_padding_inside_id(self):
        """Q38: the model put a zero-width space inside the ID and padded the brackets."""
        text = INVISIBLE.sub("", "Notify within 7 days\u202f[ parent\u200bal_leave#how-to-request ].")
        self.assertEqual(parse_citations(text, self.allowed), (["parental_leave#how-to-request"], []))

    def test_missing_document_prefix_is_flagged(self):
        """Q37: '[partner-benefit]' is not a retrieved ID and must be reported, not silently ignored."""
        valid, invalid = parse_citations("Partner leave applies [partner-benefit].", self.allowed)
        self.assertEqual((valid, invalid), ([], ["partner-benefit"]))

    def test_unretrieved_id_is_flagged(self):
        self.assertEqual(parse_citations("See [expenses#travel-and-meals].", self.allowed),
                         ([], ["expenses#travel-and-meals"]))

    def test_ordinary_brackets_are_ignored(self):
        self.assertEqual(parse_citations("Two options [1] or [see below].", self.allowed), ([], []))


if __name__ == "__main__":
    unittest.main()
