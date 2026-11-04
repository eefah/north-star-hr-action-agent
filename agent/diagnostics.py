"""Restricted diagnostic log for technical failure details.

Employees see only the fixed messages in tools.ERROR_MESSAGES. The trace (ToolResult.errors and data) gets
only a failure kind, an exception type name and a diagnostic reference, because a trace is not private
by default: whoever can see it must be access-controlled separately.

Full details (exception text and traceback) go only to the "northstar.diagnostics" logger, after
redaction. That logger never propagates to the root logger, so nothing reaches the console or the general
app logs by accident. With no handler configured, records are discarded. configure() writes them to a
file readable only by the owner (0600) in a directory only the owner can open (0700). Access to that
file is the restriction; the code cannot enforce who the owner is.
"""
import logging
import os
import re
import traceback
import uuid
from pathlib import Path

import config

LOGGER = logging.getLogger("northstar.diagnostics")
LOGGER.propagate = False
LOGGER.addHandler(logging.NullHandler())

DEFAULT_PATH = config.PROJECT_DIR / "logs" / "diagnostics.log"
SECRET_ENV_VARS = ("GROQ_API_KEY", "PINECONE_API_KEY")
REDACTED = "[REDACTED]"
_PATTERNS = [
    re.compile(r"\bgsk_[A-Za-z0-9]{8,}"),  # Groq API keys
    re.compile(r"\bpcsk_[A-Za-z0-9_\-]{8,}"),  # Pinecone API keys
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(api[-_]?key|authorization|x-api-key|token|secret|password)(\s*[=:]\s*|['\"]\s*:\s*['\"])"
               r"[^\s,'\"}]+"),
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email addresses
]


def redact(text: str) -> str:
    """Remove API keys, credentials and email addresses before anything is written to the diagnostic log."""
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, REDACTED)
    for p in _PATTERNS:
        text = p.sub(lambda m: (m.group(1) + m.group(2) + REDACTED) if m.re.groups == 2 else REDACTED, text)
    return text


def new_ref() -> str:
    return uuid.uuid4().hex[:10]


def record(ref: str, kind: str, exc: BaseException, *, tool: str, attempt: int | None = None) -> None:
    """Write one failure, redacted, to the restricted diagnostic log under the reference shown in the trace."""
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    where = f"tool={tool}" + (f" attempt={attempt}" if attempt is not None else "")
    LOGGER.error("ref=%s kind=%s %s\n%s", ref, kind, where, redact(detail))


def configure(path: Path = DEFAULT_PATH) -> logging.Handler:
    """Send diagnostic records to an owner-only file. Called by the app at start-up, never by the tools."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    os.close(os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600))
    os.chmod(path, 0o600)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return handler
