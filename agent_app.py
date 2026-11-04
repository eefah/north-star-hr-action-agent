"""Supervised demonstration of the Stage 3 HR Policy Action Agent. Synthetic employee data only.

    streamlit run agent_app.py

All logic is in agent/graph.py (the workflow) and agent/demo.py (storage and display helpers). This page
only collects input, starts or resumes a workflow and shows the employee view. Workflows are saved in a
local SQLite checkpoint, so a request can be paused and reopened after a page reload or an app restart.
"""
import uuid
from datetime import date

import streamlit as st

from agent import demo
from agent import graph as G
from agent.tools import MAX_NOTE

st.set_page_config(page_title="Northstar HR Action Agent (demo)", page_icon="🧭", layout="centered")

MARKS = {"done": "✅", "stopped": "⛔", "problem": "⚠️", "waiting": "⏸️", "not reached": "▫️", "cancelled": "✖️"}
NOT_STATED = "(not stated)"
EMPLOYMENT = {"Permanent": "permanent", "Contractor": "contractor", "Other (fixed-term, agency, intern…)": "other",
              NOT_STATED: None}
UK = {"Yes": True, "No": False, NOT_STATED: None}
PATTERN = {"Full time": "full_time", "Part time": "part_time", NOT_STATED: None}


@st.cache_resource(show_spinner="Connecting to the policy services…")
def services():
    return demo.make_services()


@st.cache_resource
def workflow_app():
    checkpointer, conn = demo.open_checkpointer()
    return G.build_graph(checkpointer), conn


def ctx():
    return demo.graph_context(services(), st.session_state.get("demo_date") or date.today())


def state_of(app, thread_id):
    return app.get_state({"configurable": {"thread_id": thread_id}}).values


def run(action, *args):
    """Start or resume a workflow. The graph never raises to the page; this is a last guard for the page."""
    app, _ = workflow_app()
    try:
        with st.spinner("Checking against the policy…"):
            return action(app, ctx(), *args)
    except ValueError:
        st.error("That workflow can't be continued. Start a new request.")


# ---------------------------------------------------------------- sidebar

def sidebar():
    app, conn = workflow_app()
    with st.sidebar:
        st.header("Demo controls")
        st.info(f"Synthetic employee **{demo.EMPLOYEE_REF}** (role: employee). Do not enter real personal "
                "information.")
        st.date_input("Request date (demo)", value=date.today(), key="demo_date", format="DD/MM/YYYY",
                      help="The date the agent treats as today. Change it to show the notice rule being "
                           "re-checked when a draft is approved on a later day.")
        if st.button("New request", key="new_request", width="stretch"):
            st.session_state.pop("thread", None)
            st.rerun()

        st.subheader("Saved workflows")
        st.caption("Workflows are saved after every step. Reload the page or restart the app, then reopen one "
                   "here to resume it.")
        flows = demo.list_workflows(app, conn)
        if flows:
            labels = {f["thread_id"]: f"{f['thread_id']} · {f['status']}" for f in flows}
            choice = st.selectbox("Workflow", list(labels), format_func=labels.get, key="workflow_choice")
            if st.button("Open workflow", key="open_workflow", width="stretch"):
                st.session_state.thread = choice
                st.rerun()
        else:
            st.caption("None yet.")

        with st.expander("Delete demonstration data"):
            st.caption("Removes every saved workflow, approved draft and diagnostic log in demo_data/.")
            confirm = st.checkbox("I understand this can't be undone", key="confirm_delete")
            if st.button("Delete all demo data", key="delete_all", disabled=not confirm):
                workflow_app.clear()
                conn.close()
                demo.delete_all_demo_data()
                st.session_state.pop("thread", None)
                st.success("All demonstration data deleted.")
                st.rerun()


# ---------------------------------------------------------------- new request forms

def new_request():
    st.title("HR Policy Action Agent")
    st.caption("Supervised demonstration with synthetic data. The agent prepares drafts for your review; it "
               "never submits anything to PeopleHub or contacts anyone.")
    kind = st.radio("What would you like to do?", ["Ask a policy question", "Request annual leave"],
                    horizontal=True, key="kind")
    if kind == "Ask a policy question":
        with st.form("question_form"):
            question = st.text_area("Your question", key="question",
                                    placeholder="How much notice do I need to give for annual leave?")
            if st.form_submit_button("Ask"):
                start("policy_question", {"question": question})
        return
    with st.form("leave_form"):
        c1, c2, c3 = st.columns(3)
        employment = c1.selectbox("Employment type", list(EMPLOYMENT), key="employment")
        uk = c2.selectbox("Based in the UK?", list(UK), key="uk")
        pattern = c3.selectbox("Working pattern", list(PATTERN), key="pattern")
        d1, d2 = st.columns(2)
        start_date = d1.text_input("First day of leave", placeholder="DD/MM/YYYY", key="start_date")
        end_date = d2.text_input("Last day of leave", placeholder="DD/MM/YYYY", key="end_date")
        note = st.text_area("Note for your manager (optional)", max_chars=MAX_NOTE, key="note",
                            help="Synthetic demo: do not include real personal or health information.")
        if st.form_submit_button("Check and prepare draft"):
            problems = demo.note_problems(note)
            if problems:
                for p in problems:
                    st.error(p)
                return
            request = {"employment_type": EMPLOYMENT[employment], "uk_based": UK[uk],
                       "working_pattern": PATTERN[pattern], "start_date": start_date, "end_date": end_date}
            if note.strip():
                request["employee_note"] = note.strip()
            start("leave_request", request)


def start(kind, request):
    thread = f"{'leave' if kind == 'leave_request' else 'question'}-{uuid.uuid4().hex[:8]}"
    if run(G.start, thread, kind, request) is not None:
        st.session_state.thread = thread
        st.rerun()


# ---------------------------------------------------------------- workflow view

def show_workflow(thread):
    app, _ = workflow_app()
    view = G.employee_view(app, thread)
    state = state_of(app, thread)
    if not state:
        st.warning("That workflow no longer exists.")
        st.session_state.pop("thread", None)
        return
    title = "Policy question" if state.get("kind") == "policy_question" else "Annual leave request"
    st.title(title)
    st.caption(f"Workflow `{thread}` · saved after every step")

    label, colour = demo.outcome_label(view)
    st.badge(label, color=colour)
    if view["message"]:
        st.markdown(view["message"])
    if view["recovery"]:
        st.caption(f"Next step: {view['recovery']}")

    with st.expander("Progress", expanded=True):
        for row in demo.progress(state):
            st.markdown(f"{MARKS[row['mark']]} {row['label']} · *{row['mark']}*")

    pending = view["pending"] or {}
    if pending.get("kind") == "employee_info":
        ask_info(thread, pending)
    elif pending.get("kind") == "retry":
        offer_retry(thread, pending)
    elif pending.get("kind") == "approval":
        approval(thread, view["draft"], pending)
    elif view["status"] == G.COMPLETED and view["draft"]:
        show_draft(view["draft"], saved=True)

    show_evidence(state)


def ask_info(thread, pending):
    if pending.get("error"):
        st.error(pending["error"])
    st.markdown(f"**{pending['question']}**")
    with st.form("info_form"):
        reply = {}
        if "employment_type" in pending["missing"]:
            reply["employment_type"] = EMPLOYMENT[st.selectbox("Employment type", list(EMPLOYMENT)[:-1],
                                                               key="info_employment")]
        if "uk_based" in pending["missing"]:
            reply["uk_based"] = UK[st.selectbox("Based in the UK?", ["Yes", "No"], key="info_uk")]
        if "working_pattern" in pending["missing"]:
            reply["working_pattern"] = PATTERN[st.selectbox("Working pattern", ["Full time", "Part time"],
                                                            key="info_pattern")]
        if st.form_submit_button("Continue"):
            run(G.resume, thread, reply)
            st.rerun()


def offer_retry(thread, pending):
    st.warning(pending["message"])
    c1, c2 = st.columns(2)
    if c1.button("Try again", key="retry_yes", type="primary"):
        run(G.resume, thread, {"retry": True})
        st.rerun()
    if c2.button("Stop here", key="retry_no"):
        run(G.resume, thread, {"retry": False})
        st.rerun()


def approval(thread, draft, pending):
    if pending.get("error"):
        st.error(pending["error"])
    show_draft(draft, saved=False)
    st.markdown(f"**{pending['question']}**")
    c1, c2 = st.columns(2)
    reply = {"draft_id": draft["draft_id"], "by": demo.EMPLOYEE_REF}
    if c1.button("Approve this draft", key="approve", type="primary"):
        run(G.resume, thread, reply | {"decision": "approve"})
        st.rerun()
    if c2.button("Reject", key="reject"):
        run(G.resume, thread, reply | {"decision": "reject"})
        st.rerun()


def show_draft(d, saved):
    st.subheader("Approved draft (saved locally)" if saved else "Draft for your review")
    st.warning(d["status_text"])
    dates = d["dates"]
    st.markdown(f"**Draft** `{d['draft_id']}` · prepared on {d['prepared_on']}  \n"
                f"**Leave:** {dates['start']} to {dates['end']} · first day of leave {dates['first_leave_day']}  \n"
                f"**Working days requested:** {dates['working_days_requested']}  \n"
                f"*{dates['working_days_note']}*  \n"
                f"**Request must be submitted by:** {d['notice']['latest_request_date']} "
                f"(`{d['notice']['citation']}`)")
    st.markdown("**Approvals needed after you submit it in PeopleHub:**")
    for a in d["approvals_required"]:
        st.markdown(f"- {a['approver']}: {a['status']} · “{a['basis']}” `{a['citation']}`")
    for note in d["eligibility_notes"] + d["flags"]:
        st.markdown(f"- {note}")
    if d.get("employee_note"):
        st.markdown(f"**Your note:** {d['employee_note']}")
    with st.expander("Assumptions (not policy)"):
        for a in d["assumptions"]:
            st.caption(a)


def show_evidence(state):
    _, retriever, _ = services()
    passages = demo.evidence_passages(state, retriever.by_id)
    if not passages:
        return
    with st.expander(f"Policy evidence used ({len(passages)})"):
        for p in passages:
            mark = " · **cited**" if p["cited"] else ""
            st.markdown(f"**`{p['chunk_id']}`** · {p['title']} › {p['heading']} · v{p['version']} · "
                        f"{p['status']}{mark}")
            st.caption(p["passage"])


# ---------------------------------------------------------------- page

sidebar()
if st.session_state.get("thread"):
    show_workflow(st.session_state.thread)
else:
    new_request()
