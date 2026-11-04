"""HR Policy Action Agent (Week 3): helps an employee prepare an annual leave request for their own review.

dates.py  calendar arithmetic only, no HR policy
rules.py  annual leave rules, each tied to a verbatim policy quote and its chunk ID
tools.py  the RAG pipeline and the rules as validated tools with machine-readable result codes
diagnostics.py  restricted, redacted log for technical failure details
explain.py  result code -> can_proceed / cannot_proceed / cannot_determine, with the employee explanation
graph.py  Stage 3 LangGraph workflow: checkpointed, resumable, employee approval before any save
"""
