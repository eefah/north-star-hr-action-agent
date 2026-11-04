"""Support for the supervised Streamlit demonstration (agent_app.py). Synthetic employee data only.

Storage. Everything the demo writes (checkpoints, approved drafts, diagnostic logs) lives under one local,
Git-ignored folder, demo_data/, created owner-only (0700) with owner-only files (0600) where the operating
system supports it. delete_all_demo_data() removes it in one step; from a terminal:

    python -m agent.demo --delete-all

This is a demonstration, not production-grade privacy compliance: there is no encryption at rest, no
retention schedule beyond manual deletion, and no access control beyond file permissions.

Synthetic data. The demo employee is fixed (EMPLOYEE_REF). Notes are optional, at most 300 characters,
and refused if they look like real personal information (email addresses, phone numbers, National
Insurance numbers). Notes are stored only in the local checkpoint and the approved draft; they are never
written to the diagnostic log or the trace.
"""
import argparse
import os
import re
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import config
from agent import diagnostics
from agent import graph as G
from agent.tools import MAX_NOTE

DEMO_DIR = config.PROJECT_DIR / "demo_data"
EMPLOYEE_REF = "synthetic-employee-01"
ROLES = ("employee",)

_PERSONAL = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "an email address"),
    (re.compile(r"(?:\+?\d[\s\-()]*){9,}"), "a phone number or long number"),
    (re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I), "a National Insurance number"),
]


# ---------------------------------------------------------------- storage

def paths(base: Path | None = None) -> dict[str, Path]:
    base = base or DEMO_DIR
    return {"base": base, "checkpoints": base / "checkpoints.sqlite", "drafts": base / "drafts",
            "log": base / "logs" / "diagnostics.log"}


def prepare_storage(base: Path | None = None) -> dict[str, Path]:
    """Create the owner-only demo folders and turn on the restricted diagnostic log (once per process)."""
    p = paths(base)
    for folder in (p["base"], p["drafts"], p["log"].parent):
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
    if not any(getattr(h, "baseFilename", None) == str(p["log"]) for h in diagnostics.LOGGER.handlers):
        diagnostics.configure(p["log"])
    return p


def open_checkpointer(base: Path | None = None):
    """The SQLite checkpointer, in an owner-only file. Returns (checkpointer, connection)."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    p = prepare_storage(base)
    os.close(os.open(p["checkpoints"], os.O_CREAT | os.O_WRONLY, 0o600))
    os.chmod(p["checkpoints"], 0o600)
    conn = sqlite3.connect(p["checkpoints"], check_same_thread=False)
    return SqliteSaver(conn), conn


def list_workflows(app, conn) -> list[dict]:
    """Saved workflows, newest first, with their status, for the 'resume' list."""
    try:
        rows = conn.execute("select thread_id, max(checkpoint_id) from checkpoints group by thread_id "
                            "order by 2 desc").fetchall()
    except sqlite3.OperationalError:  # no workflow saved yet
        return []
    out = []
    for thread_id, _ in rows:
        s = app.get_state({"configurable": {"thread_id": thread_id}}).values
        out.append({"thread_id": thread_id, "kind": s.get("kind"), "status": s.get("status")})
    return out


def delete_all_demo_data(base: Path | None = None) -> bool:
    """Remove every checkpoint, draft and log the demo has written. Returns True if anything was deleted."""
    p = paths(base)
    for h in list(diagnostics.LOGGER.handlers):  # release the log file before deleting it
        if str(getattr(h, "baseFilename", "")).startswith(str(p["base"])):
            diagnostics.LOGGER.removeHandler(h)
            h.close()
    if not p["base"].exists():
        return False
    shutil.rmtree(p["base"])
    return True


# ---------------------------------------------------------------- synthetic data guard

def note_problems(note: str) -> list[str]:
    """Why a note can't be accepted in the demo. Empty list means it is fine (an empty note is fine)."""
    problems = []
    if len(note) > MAX_NOTE:
        problems.append(f"The note is longer than {MAX_NOTE} characters.")
    for pattern, what in _PERSONAL:
        if pattern.search(note):
            problems.append(f"The note seems to contain {what}. This demo uses synthetic data only: please remove "
                            "any real personal information.")
    return problems


# ---------------------------------------------------------------- display helpers

OUTCOME_LABELS = {  # (label, colour) for st.badge
    "can_proceed": ("Can proceed", "green"),
    "cannot_proceed": ("Cannot proceed", "red"),
    "cannot_determine": ("Cannot determine", "orange"),
    "cancelled": ("Cancelled", "gray"),
}
WAITING_LABELS = {
    G.AWAITING_INFO: ("Waiting for your details", "blue"),
    G.AWAITING_RETRY: ("Cannot determine yet: temporary problem", "orange"),
}
LEAVE_STEPS = [("intake", "Request received"), ("eligibility", "Eligibility checked"),
               ("ask_info", "Details requested"), ("dates", "Dates and notice checked"),
               ("draft", "Draft prepared"), ("approval", "Your approval"), ("save_draft", "Approved draft saved")]
QUESTION_STEPS = [("intake", "Question received"), ("answer", "Policy answer checked")]


def outcome_label(view: dict) -> tuple[str, str]:
    if view.get("status") in WAITING_LABELS:
        return WAITING_LABELS[view["status"]]
    return OUTCOME_LABELS.get(view.get("category"), ("In progress", "gray"))


def progress(state: dict) -> list[dict]:
    """One row per workflow step: done, stopped (rule), problem (cannot determine), waiting or not reached.
    Built from the trace, but shows only step names and outcomes, never errors or references."""
    steps = QUESTION_STEPS if state.get("kind") == "policy_question" else LEAVE_STEPS
    last = {}
    for e in state.get("trace", []):
        last[e["step"]] = e
    status, outcome = state.get("status"), state.get("outcome") or {}
    waiting_step = {G.AWAITING_INFO: "ask_info", G.AWAITING_APPROVAL: "approval",
                    G.AWAITING_RETRY: state.get("retry_target")}.get(status)
    rows = []
    for step, label in steps:
        e = last.get(step)
        if step == waiting_step:
            mark = "waiting"
        elif e is None:
            if step == "ask_info":
                continue  # only shown when it happened
            mark = "not reached"
        elif e.get("code") in (None, "OK") and e.get("decision") != "reject":
            mark = "done"
        elif step == "approval" and e.get("decision") == "reject":
            mark = "cancelled"
        else:
            category = outcome.get("category") if outcome.get("code") == e.get("reason", e.get("code")) else None
            mark = "stopped" if category == "cannot_proceed" else "problem"
        rows.append({"step": step, "label": label, "mark": mark})
    return rows


def evidence_passages(state: dict, chunks_by_id: dict) -> list[dict]:
    """The verified policy passages the decisions relied on, for display, marked if cited in the outcome."""
    cited = set((state.get("outcome") or {}).get("citations", []))
    out = []
    for ref in state.get("evidence_refs", []):
        chunk = chunks_by_id.get(ref["chunk_id"], {})
        text = chunk.get(config.TEXT_FIELD, "")
        out.append({**ref, "cited": ref["chunk_id"] in cited, "heading": chunk.get("heading"),
                    "passage": text.split("\n\n", 1)[-1]})
    return out


def make_services():
    """Live clients (Pinecone, reranker, Groq) and the bank holiday data. Patched in interface tests."""
    from agent.dates import BankHolidays
    from rag import RAG
    rag = RAG()
    return rag, rag.retriever, BankHolidays.load()


def graph_context(services, today: date, base: Path | None = None) -> G.GraphContext:
    from agent.tools import ToolContext
    rag, retriever, holidays = services
    tools = ToolContext(roles=ROLES, today=today, retriever=retriever, rag=rag, holidays=holidays,
                        employee_ref=EMPLOYEE_REF)
    return G.GraphContext(tools, paths(base)["drafts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the local demonstration data.")
    parser.add_argument("--delete-all", action="store_true", help="delete all checkpoints, drafts and logs")
    if parser.parse_args().delete_all:
        print("Deleted all demonstration data." if delete_all_demo_data() else "There was no demonstration data.")
    else:
        parser.print_help()
