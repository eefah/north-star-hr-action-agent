"""Calendar arithmetic for leave requests.

No HR policy lives here. The rules that use these functions (the notice period, the approval threshold)
are in agent/rules.py, each with its policy citation. What counts as a working day is an assumption,
because the annual leave policy doesn't define it, so it is stated in WORKING_DAY_ASSUMPTION and shown
to the employee wherever a working-day count is used.
"""
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import config

BANK_HOLIDAYS_FILE = config.PROJECT_DIR / "data" / "reference" / "bank_holidays_england_wales.json"

WORKING_DAY_ASSUMPTION = (
    "Assumption, not policy: a working day is Monday to Friday, excluding England and Wales bank holidays. "
    "The annual leave policy does not define 'working day'."
)


class CalendarError(ValueError):
    """A date the calendar can't handle reliably, e.g. a year the bank holiday data doesn't cover."""


@dataclass(frozen=True)
class BankHolidays:
    dates: frozenset[date]
    first_year: int
    last_year: int
    source: str

    @classmethod
    def load(cls, path: Path = BANK_HOLIDAYS_FILE) -> "BankHolidays":
        data = json.loads(path.read_text(encoding="utf-8"))
        first, last = data["covers_years"]
        return cls(frozenset(date.fromisoformat(e["date"]) for e in data["events"]), first, last, data["source"])

    def check_covers(self, d: date) -> None:
        """Refuse dates outside the data rather than silently treating them as having no bank holidays."""
        if not self.first_year <= d.year <= self.last_year:
            raise CalendarError(f"{d.isoformat()} is outside the bank holiday data, which covers "
                                f"{self.first_year} to {self.last_year} only")


DATE_FORMAT_HINT = "DD/MM/YYYY, day first (UK format), for example 23/11/2026"
_UK_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_date(text: str) -> date:
    """A date in UK format DD/MM/YYYY (always day first, never US month first), or ISO YYYY-MM-DD.
    The format is checked before anything is interpreted, so forms like 20261123, 2026-W48-1, 23/11/26
    or 23.11.2026 are refused rather than guessed. Relative phrases like 'next Friday' are resolved by the
    agent and confirmed with the employee, not here."""
    if not isinstance(text, str):
        raise ValueError(f"not a date: {text!r}. Use {DATE_FORMAT_HINT}")
    s = text.strip()
    try:
        if m := _UK_DATE.fullmatch(s):
            day, month, year = map(int, m.groups())
            return date(year, month, day)
        if m := _ISO_DATE.fullmatch(s):
            year, month, day = map(int, m.groups())
            return date(year, month, day)
    except ValueError:
        raise ValueError(f"{text!r} is not a real calendar date. Use {DATE_FORMAT_HINT}") from None
    raise ValueError(f"{text!r} is not in a recognised date format. Use {DATE_FORMAT_HINT}")


parse_iso_date = parse_date  # earlier name, kept for existing callers


def is_working_day(d: date, holidays: BankHolidays) -> bool:
    holidays.check_covers(d)
    return d.weekday() < 5 and d not in holidays.dates


def working_days(start: date, end: date, holidays: BankHolidays) -> list[date]:
    """Working days from start to end, both included."""
    if end < start:
        raise ValueError(f"end date {end.isoformat()} is before start date {start.isoformat()}")
    days = (start + timedelta(n) for n in range((end - start).days + 1))
    return [d for d in days if is_working_day(d, holidays)]


def non_working_days(start: date, end: date, holidays: BankHolidays) -> list[date]:
    """Weekends and bank holidays from start to end, both included."""
    wd = set(working_days(start, end, holidays))
    return [start + timedelta(n) for n in range((end - start).days + 1) if start + timedelta(n) not in wd]


def shift_working_days(d: date, n: int, holidays: BankHolidays) -> date:
    """The nth working day after d (n > 0) or before d (n < 0). d itself is never counted."""
    if n == 0:
        raise ValueError("n must not be 0")
    step = timedelta(1 if n > 0 else -1)
    remaining = abs(n)
    while remaining:
        d += step
        remaining -= is_working_day(d, holidays)
    return d
