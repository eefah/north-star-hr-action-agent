"""Annual leave rules (agent/rules.py). Offline: evidence is built from data/annual_leave.md with the same
build_chunks() the ingestion script uses, so no Pinecone index is needed. All employees are synthetic.

    python -m unittest discover tests
"""
import copy
import unittest
from datetime import date

import config
from agent.dates import WORKING_DAY_ASSUMPTION, BankHolidays
from agent.rules import HEAD_OF_PEOPLE, LINE_MANAGER, RULES, EmployeeProfile, assess_dates, check_eligibility, \
    verify_evidence, ELIGIBILITY_RULES
from ingest import build_chunks

BH = BankHolidays.load()
POLICY = config.DATA_DIR / "annual_leave.md"
EVIDENCE = {c["_id"]: c for c in build_chunks([POLICY])}
TODAY = date(2026, 11, 3)  # Tuesday; v2.0 of the policy is in force from 1 Oct 2026

FULL_TIME = EmployeeProfile("permanent", True, "full_time")
PART_TIME = EmployeeProfile("permanent", True, "part_time")
CONTRACTOR = EmployeeProfile("contractor")


def evidence_without(chunk_id):
    return {k: v for k, v in EVIDENCE.items() if k != chunk_id}


def evidence_with(chunk_id, **changes):
    ev = copy.deepcopy(EVIDENCE)
    ev[chunk_id].update(changes)
    return ev


class PolicyGroundingTests(unittest.TestCase):
    def test_every_rule_is_quoted_verbatim_from_its_policy_section(self):
        """Guards against invented rules: each quote must exist word for word in the chunk it cites."""
        for name, rule in RULES.items():
            with self.subTest(rule=name):
                self.assertIn(rule.chunk_id, EVIDENCE)
                self.assertIn(rule.quote, EVIDENCE[rule.chunk_id][config.TEXT_FIELD])

    def test_full_evidence_verifies(self):
        check = verify_evidence(EVIDENCE, list(RULES))
        self.assertTrue(check.ok, check.problems)
        self.assertEqual(set(check.citations), {"annual_leave#eligibility", "annual_leave#how-to-request",
                                                "annual_leave#contact"})

    def test_missing_wrong_version_expired_or_reworded_evidence_is_rejected(self):
        cases = {
            "missing": evidence_without("annual_leave#eligibility"),
            "version": evidence_with("annual_leave#eligibility", version="3.0"),
            "expired": evidence_with("annual_leave#eligibility", status="expired"),
            "reworded": evidence_with("annual_leave#eligibility", chunk_text="Annual leave > Eligibility\n\n"
                                      "Permanent UK employees receive 28 days of paid annual leave."),
        }
        for name, ev in cases.items():
            with self.subTest(case=name):
                check = verify_evidence(ev, ELIGIBILITY_RULES)
                self.assertFalse(check.ok)
                self.assertTrue(any("annual_leave#eligibility" in p for p in check.problems))


class EligibilityTests(unittest.TestCase):
    def test_permanent_uk_full_time_is_eligible(self):
        r = check_eligibility(FULL_TIME, EVIDENCE)
        self.assertEqual(r.status, "eligible")
        self.assertIn(RULES["eligible_group"].quote, r.notes)
        self.assertIn("annual_leave#eligibility", r.citations)

    def test_part_time_entitlement_is_not_calculated(self):
        r = check_eligibility(PART_TIME, EVIDENCE)
        self.assertEqual(r.status, "eligible")
        notes = " ".join(r.notes)
        self.assertIn("does not calculate", notes)
        self.assertNotIn("25 days", notes)

    def test_no_balance_is_ever_stated(self):
        notes = " ".join(check_eligibility(FULL_TIME, EVIDENCE).notes)
        self.assertIn("can't check", notes)
        self.assertNotRegex(notes, r"\b(remaining|left|balance)\b[^.]*\d")

    def test_contractor_is_not_eligible_without_further_questions(self):
        r = check_eligibility(CONTRACTOR, EVIDENCE)
        self.assertEqual((r.status, r.missing), ("not_eligible", []))
        self.assertIn(RULES["contractors"].quote, r.reasons)

    def test_missing_information_is_listed(self):
        self.assertEqual(check_eligibility(EmployeeProfile(), EVIDENCE).missing,
                         ["employment_type", "uk_based", "working_pattern"])
        r = check_eligibility(EmployeeProfile("permanent"), EVIDENCE)
        self.assertEqual((r.status, r.missing), ("needs_info", ["uk_based", "working_pattern"]))

    def test_cases_the_policy_does_not_cover_go_to_the_people_team(self):
        for profile in (EmployeeProfile("other", True, "full_time"), EmployeeProfile("permanent", False, "full_time")):
            with self.subTest(profile=profile):
                r = check_eligibility(profile, EVIDENCE)
                self.assertEqual(r.status, "cannot_determine")
                self.assertIn("People team", " ".join(r.reasons))

    def test_no_decision_without_evidence(self):
        r = check_eligibility(FULL_TIME, evidence_without("annual_leave#eligibility"))
        self.assertEqual(r.status, "cannot_determine")
        self.assertEqual(r.citations, [])


class DateAssessmentTests(unittest.TestCase):
    def assess(self, start, end, today=TODAY, evidence=EVIDENCE):
        return assess_dates(date.fromisoformat(start), date.fromisoformat(end), today, BH, evidence)

    def test_three_days_with_enough_notice(self):
        a = self.assess("2026-11-23", "2026-11-25")
        self.assertEqual((a.status, a.working_day_count, a.approvers), ("ok", 3, [LINE_MANAGER]))
        self.assertIn("annual_leave#how-to-request", a.citations)
        self.assertIn(WORKING_DAY_ASSUMPTION, a.assumptions)  # assumptions are always shown, never hidden

    def test_short_notice_offers_the_earliest_valid_start(self):
        a = self.assess("2026-11-10", "2026-11-12")
        self.assertEqual(a.status, "needs_change")
        self.assertEqual(a.latest_request_date, date(2026, 10, 27))
        self.assertEqual(a.earliest_valid_start, date(2026, 11, 17))
        self.assertIn(RULES["notice"].quote, a.reasons[0])

    def test_notice_boundary(self):
        self.assertEqual(self.assess("2026-11-17", "2026-11-17").status, "ok")  # exactly 10 working days
        self.assertEqual(self.assess("2026-11-16", "2026-11-16").status, "needs_change")  # 9

    def test_request_made_at_the_weekend(self):
        a = self.assess("2026-11-20", "2026-11-20", today=date(2026, 11, 7))  # Saturday
        self.assertEqual((a.status, a.earliest_valid_start), ("needs_change", date(2026, 11, 23)))

    def test_head_of_people_needed_above_10_working_days(self):
        ten = self.assess("2026-11-30", "2026-12-11")
        eleven = self.assess("2026-11-30", "2026-12-14")
        self.assertEqual((ten.working_day_count, ten.approvers), (10, [LINE_MANAGER]))
        self.assertEqual((eleven.working_day_count, eleven.approvers), (11, [LINE_MANAGER, HEAD_OF_PEOPLE]))
        self.assertIn(RULES["long_leave"].quote, eleven.reasons)

    def test_christmas_range_flags_bank_holidays_and_two_leave_years(self):
        a = self.assess("2026-12-21", "2027-01-01")
        self.assertEqual((a.status, a.working_day_count), ("ok", 7))
        flags = " ".join(a.flags)
        self.assertIn("2026-12-25, 2026-12-28, 2027-01-01", flags)
        self.assertIn("spans two leave years", flags)
        self.assertTrue(any(s.startswith("Assumption, not policy") for s in a.assumptions))

    def test_weekend_start_is_flagged(self):
        a = self.assess("2026-11-21", "2026-11-25")
        self.assertEqual((a.first_leave_day, a.working_day_count), (date(2026, 11, 23), 3))
        self.assertIn("not a working day", " ".join(a.flags))

    def test_invalid_ranges(self):
        cases = {"end before start": ("2026-11-25", "2026-11-23"), "in the past": ("2026-11-02", "2026-11-04"),
                 "weekend only": ("2026-11-21", "2026-11-22"), "outside data": ("2029-01-08", "2029-01-09")}
        for name, (s, e) in cases.items():
            with self.subTest(case=name):
                a = self.assess(s, e)
                self.assertEqual(a.status, "invalid")
                self.assertEqual(a.approvers, [])

    def test_leave_starting_today_needs_change_not_invalid(self):
        self.assertEqual(self.assess("2026-11-03", "2026-11-03").status, "needs_change")

    def test_no_assessment_without_evidence(self):
        a = self.assess("2026-11-23", "2026-11-25", evidence=evidence_without("annual_leave#how-to-request"))
        self.assertEqual((a.status, a.approvers, a.citations), ("cannot_determine", [], []))


if __name__ == "__main__":
    unittest.main()
