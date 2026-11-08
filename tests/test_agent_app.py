"""Streamlit demo (agent_app.py) and its helpers (agent/demo.py). Offline: fake Pinecone and Groq, and a
temporary demo folder. Synthetic data only.

    python -m unittest discover tests
"""
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from agent import demo, diagnostics
from agent import graph as G
from tests.fakes import HOLIDAYS, TODAY, FakeLLM, make_rag, make_retriever

APP = str(config.PROJECT_DIR / "agent_app.py")
GOOD_ANSWER = "Permanent UK employees get 25 days [annual_leave#eligibility]."


def fake_services(llm_text=GOOD_ANSWER, **retriever_kwargs):
    retriever = make_retriever(**retriever_kwargs)
    return lambda: (make_rag(retriever, FakeLLM(llm_text)), retriever, HOLIDAYS)


def page_text(at) -> str:
    parts = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "error", "warning", "info", "success",
                 "exception"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.base = self.tmp / "demo_data"
        st.cache_resource.clear()
        self.patches = [mock.patch.object(demo, "DEMO_DIR", self.base)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        st.cache_resource.clear()
        for p in self.patches:
            p.stop()
        demo.delete_all_demo_data(self.base)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def services(self, factory):
        p = mock.patch.object(demo, "make_services", factory)
        p.start()
        self.patches.append(p)

    def app(self):
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        at.date_input(key="demo_date").set_value(TODAY).run()
        return at

    def submit_leave(self, at, start="23/11/2026", end="25/11/2026", employment="Permanent", note=""):
        at.radio(key="kind").set_value("Request annual leave").run()
        at.selectbox(key="employment").set_value(employment)
        at.text_input(key="start_date").set_value(start)
        at.text_input(key="end_date").set_value(end)
        at.text_area(key="note").set_value(note)
        at.button[[b.label for b in at.button].index("Check and prepare draft")].click().run()
        return at

    def drafts(self):
        d = self.base / "drafts"
        return sorted(d.glob("*.json")) if d.exists() else []

    def assertNoTechnicalDetail(self, at):
        text = page_text(at)
        for leak in ("Traceback", "ConnectionError", "diagnostic ref", "pcsk_", "Exception"):
            self.assertNotIn(leak, text)
        self.assertEqual(len(at.exception), 0)


class JourneyTests(AppTestCase):
    def test_leave_request_is_reviewed_approved_and_saved(self):
        self.services(fake_services())
        at = self.submit_leave(self.app())
        text = page_text(at)
        self.assertIn("Can proceed", text)
        self.assertIn("Draft for your review", text)
        self.assertIn("NOT been submitted to PeopleHub", text)
        self.assertTrue(any(e.label.startswith("Policy evidence used") for e in at.expander))
        self.assertEqual(self.drafts(), [])  # nothing saved before approval

        at.button(key="approve").click().run()
        text = page_text(at)
        self.assertIn("Approved draft (saved locally)", text)
        self.assertIn("Approved draft saved · *done*", text)
        self.assertEqual(len(self.drafts()), 1)
        self.assertNoTechnicalDetail(at)

    def test_policy_question_shows_answer_and_cited_evidence(self):
        self.services(fake_services())
        at = self.app()
        at.text_area(key="question").set_value("How many annual leave days do permanent UK employees get?")
        at.button[[b.label for b in at.button].index("Ask")].click().run()
        text = page_text(at)
        self.assertIn("Can proceed", text)
        self.assertIn("25 days", text)
        self.assertIn("annual_leave#eligibility", text)
        self.assertIn("**cited**", text)

    def test_three_outcomes_are_labelled(self):
        self.services(fake_services())
        refused = self.submit_leave(self.app(), employment="Contractor")
        self.assertIn("Cannot proceed", page_text(refused))
        self.assertIn("Contractors are outside this policy.", page_text(refused))

        st.cache_resource.clear()
        undetermined = self.submit_leave(self.app(), employment="Other (fixed-term, agency, intern…)")
        self.assertIn("Cannot determine", page_text(undetermined))
        self.assertNotIn("Cannot proceed", page_text(undetermined))

    def test_missing_details_are_asked_for(self):
        self.services(fake_services())
        at = self.app()
        at.radio(key="kind").set_value("Request annual leave").run()
        at.selectbox(key="uk").set_value("(not stated)")
        at.text_input(key="start_date").set_value("23/11/2026")
        at.text_input(key="end_date").set_value("25/11/2026")
        at.button[[b.label for b in at.button].index("Check and prepare draft")].click().run()
        self.assertIn("Waiting for your details", page_text(at))
        at.button[[b.label for b in at.button].index("Continue")].click().run()
        self.assertIn("Draft for your review", page_text(at))


class DemoDateTests(AppTestCase):
    def test_sidebar_shows_the_date_the_agent_uses(self):
        self.services(fake_services())
        at = self.app()  # demo date set to TODAY (03/11/2026)
        self.assertIn("The agent is treating today as: **03/11/2026**", page_text(at))
        self.assertIn("This is not today's real date.", page_text(at))
        at = self.submit_leave(at)  # the draft records the same date
        self.assertIn("prepared on 2026-11-03", page_text(at))


class FailureTests(AppTestCase):
    def test_temporary_fault_offers_retry_then_stops_cleanly(self):
        self.services(fake_services(fail_times=99))
        at = self.submit_leave(self.app())
        self.assertIn("Cannot determine yet: temporary problem", page_text(at))
        self.assertIn("Something went wrong while checking your request.", page_text(at))
        at.button(key="retry_no").click().run()
        text = page_text(at)
        self.assertIn("Cannot determine", text)
        self.assertIn("No action has been taken.", text)
        self.assertNoTechnicalDetail(at)
        self.assertEqual(self.drafts(), [])

    def test_retry_succeeds_after_the_service_recovers(self):
        self.services(fake_services(fail_times=3))
        at = self.submit_leave(self.app())
        at.button(key="retry_yes").click().run()
        self.assertIn("Draft for your review", page_text(at))

    def test_note_with_personal_information_is_refused(self):
        self.services(fake_services())
        at = self.submit_leave(self.app(), note="Call me on 07700 900123 or jane@example.com")
        errors = [e.value for e in at.error]
        self.assertTrue(any("email address" in e for e in errors))
        self.assertTrue(any("phone number" in e for e in errors))
        self.assertNotIn("thread", at.session_state)


class ResumeTests(AppTestCase):
    def test_workflow_resumes_in_a_new_session_after_a_restart(self):
        self.services(fake_services())
        first = self.submit_leave(self.app())
        thread = first.session_state["thread"]
        self.assertIn("Draft for your review", page_text(first))

        st.cache_resource.clear()  # simulated app restart: new clients and a new checkpoint connection
        second = self.app()
        self.assertNotIn("thread", second.session_state)
        second.selectbox(key="workflow_choice").set_value(thread)
        second.button(key="open_workflow").click().run()
        self.assertIn("Draft for your review", page_text(second))
        second.button(key="approve").click().run()
        self.assertIn("Approved draft (saved locally)", page_text(second))
        self.assertEqual(len(self.drafts()), 1)


class DemoDataTests(AppTestCase):
    def test_delete_all_removes_every_demo_file(self):
        self.services(fake_services())
        at = self.submit_leave(self.app())
        at.button(key="approve").click().run()
        self.assertEqual(len(self.drafts()), 1)
        log = demo.paths(self.base)["log"]
        at.checkbox(key="confirm_delete").check().run()
        at.button(key="delete_all").click().run()
        # The page reopens an empty store after deleting; nothing from before survives.
        self.assertEqual(self.drafts(), [])
        self.assertIn("None yet.", page_text(at))
        self.assertNotIn("thread", at.session_state)
        self.assertFalse(log.exists() and log.read_text())

    def test_storage_is_owner_only(self):
        _, conn = demo.open_checkpointer(self.base)
        conn.close()
        p = demo.paths(self.base)
        for folder in (p["base"], p["drafts"], p["log"].parent):
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700, folder)
        for f in (p["checkpoints"], p["log"]):
            self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600, f)

    def test_demo_data_is_git_ignored(self):
        ignored = (config.PROJECT_DIR / ".gitignore").read_text().split()
        self.assertIn("demo_data/", ignored)
        self.assertEqual(Path(demo.paths()["base"]).name, "demo_data")

    def test_delete_when_nothing_exists(self):
        self.assertFalse(demo.delete_all_demo_data(self.tmp / "nothing-here"))


class HelperTests(unittest.TestCase):
    def test_note_guard(self):
        self.assertEqual(demo.note_problems(""), [])
        self.assertEqual(demo.note_problems("Family holiday, back on 26/11/2026."), [])
        for note in ("jane.doe@example.com", "+44 7700 900123", "NI: AB 12 34 56 C", "x" * 301):
            with self.subTest(note=note):
                self.assertTrue(demo.note_problems(note))

    def test_raw_notes_never_reach_the_diagnostic_log_or_trace(self):
        from tests.fakes import make_ctx
        note = "Synthetic note XYZZY-NOTE"
        tools, _ = make_ctx(retriever=make_retriever(fail_times=99))
        from langgraph.checkpoint.memory import InMemorySaver
        app = G.build_graph(InMemorySaver())
        with tempfile.TemporaryDirectory() as tmp, self.assertLogs("northstar.diagnostics") as logs:
            G.start(app, G.GraphContext(tools, Path(tmp)), "n1", "leave_request",
                    {"employment_type": "permanent", "uk_based": True, "working_pattern": "full_time",
                     "start_date": "23/11/2026", "end_date": "25/11/2026", "employee_note": note})
        self.assertNotIn("XYZZY", "\n".join(logs.output))
        trace = app.get_state({"configurable": {"thread_id": "n1"}}).values["trace"]
        self.assertNotIn("XYZZY", str(trace))

    def test_labels_cover_every_status(self):
        for status, category in ((G.AWAITING_APPROVAL, "can_proceed"), (G.COMPLETED, "can_proceed"),
                                 (G.ANSWERED, "can_proceed"), (G.REFUSED, "cannot_proceed"),
                                 (G.UNDETERMINED, "cannot_determine"), (G.CANCELLED, "cancelled"),
                                 (G.AWAITING_INFO, None), (G.AWAITING_RETRY, None)):
            with self.subTest(status=status):
                label, _ = demo.outcome_label({"status": status, "category": category})
                self.assertNotEqual(label, "In progress")


if __name__ == "__main__":
    unittest.main()
