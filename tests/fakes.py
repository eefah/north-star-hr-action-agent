"""Offline stand-ins for Pinecone and Groq, shared by the agent tests.

The fake index deliberately ignores the metadata filter, so tests prove the local permission checks
work on their own. Fakes can be told to fail a number of times to exercise retries.
"""
import copy
from datetime import date
from types import SimpleNamespace

import config
from agent.dates import BankHolidays
from agent.tools import RetryPolicy, ToolContext
from ingest import build_chunks
from rag import RAG
from retriever import HybridRetriever, tokenize

ALL_CHUNKS = build_chunks(sorted(config.DATA_DIR.glob("*.md")) + sorted(config.EXPIRED_DIR.glob("*.md")))
TODAY = date(2026, 11, 3)  # a Tuesday; annual leave policy v2.0 is in force from 1 Oct 2026
HOLIDAYS = BankHolidays.load()


def _overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    return len(q & set(tokenize(text))) / len(q) if q else 0.0


class FakeIndex:
    """Pinecone search stand-in: ranks every chunk by word overlap and ignores the filter."""
    def __init__(self, chunks, fail_times=0):
        self.chunks, self.fail_times, self.calls = chunks, fail_times, 0

    def search(self, namespace, top_k, inputs, filter, fields):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("Pinecone unavailable (simulated)")
        ranked = sorted(self.chunks, key=lambda c: _overlap(inputs["text"], c[config.TEXT_FIELD]), reverse=True)
        return SimpleNamespace(result=SimpleNamespace(hits=[SimpleNamespace(id=c["_id"]) for c in ranked[:top_k]]))


class FakeReranker:
    """Pinecone rerank stand-in: word overlap as the score, so unrelated chunks score 0."""
    def rerank(self, model, query, documents, return_documents):
        return SimpleNamespace(data=[SimpleNamespace(index=i, score=_overlap(query, d))
                                     for i, d in enumerate(documents)])


def make_retriever(chunks=None, fail_times=0):
    chunks = copy.deepcopy(ALL_CHUNKS if chunks is None else chunks)
    r = HybridRetriever.__new__(HybridRetriever)  # no Pinecone client, no chunks file
    r.chunks, r.by_id = chunks, {c["_id"]: c for c in chunks}
    r.index, r.inference = FakeIndex(chunks, fail_times), FakeReranker()
    r.calls = []  # (query, roles) for every search, to check what the tools passed
    original = r.search

    def spy(query, roles, *args, **kwargs):
        r.calls.append((query, tuple(roles)))
        return original(query, roles, *args, **kwargs)
    r.search = spy
    return r


class FakeLLM:
    """Groq chat-completions stand-in returning fixed text."""
    def __init__(self, text="", fail_times=0):
        self.text, self.fail_times, self.calls = text, fail_times, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.fail_times:
            raise TimeoutError("Groq timed out (simulated)")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))])


def make_rag(retriever, llm):
    rag = RAG.__new__(RAG)  # no network clients
    rag.retriever, rag.llm = retriever, llm
    return rag


def make_ctx(roles=("employee",), retriever=None, llm=None, **kwargs):
    retriever = retriever or make_retriever()
    sleeps = []
    ctx = ToolContext(roles=roles, today=kwargs.pop("today", TODAY), retriever=retriever,
                      rag=make_rag(retriever, llm or FakeLLM()), holidays=HOLIDAYS,
                      retry=RetryPolicy(3, (0.5, 1.5)), sleep=sleeps.append, new_id=lambda: "draft-test-1",
                      **kwargs)
    return ctx, sleeps


def with_chunk(chunk_id, **changes):
    """All chunks, with one changed: for stale, reworded or tampered evidence."""
    chunks = copy.deepcopy(ALL_CHUNKS)
    next(c for c in chunks if c["_id"] == chunk_id).update(changes)
    return chunks


def without_chunk(chunk_id):
    return [c for c in copy.deepcopy(ALL_CHUNKS) if c["_id"] != chunk_id]
