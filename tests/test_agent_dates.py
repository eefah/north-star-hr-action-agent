"""Calendar arithmetic (agent/dates.py). Offline. Run with: python -m unittest discover tests"""
import unittest
from datetime import date

from agent.dates import BankHolidays, CalendarError, is_working_day, non_working_days, parse_date, parse_iso_date, \
    shift_working_days, working_days

BH = BankHolidays.load()


class BankHolidayDataTests(unittest.TestCase):
    def test_reference_file(self):
        self.assertEqual((BH.first_year, BH.last_year), (2026, 2028))
        self.assertEqual(len(BH.dates), 24)  # 8 England and Wales bank holidays a year
        self.assertTrue(BH.source.startswith("https://www.gov.uk/"))
        self.assertIn(date(2026, 12, 28), BH.dates)  # substitute Boxing Day: 26 Dec 2026 is a Saturday

    def test_dates_outside_the_data_are_refused_not_guessed(self):
        for d in (date(2025, 12, 31), date(2029, 1, 2)):
            with self.assertRaises(CalendarError):
                is_working_day(d, BH)


class WorkingDayTests(unittest.TestCase):
    def test_weekdays_weekends_and_bank_holidays(self):
        self.assertTrue(is_working_day(date(2026, 11, 23), BH))  # Monday
        self.assertFalse(is_working_day(date(2026, 11, 21), BH))  # Saturday
        self.assertFalse(is_working_day(date(2026, 12, 25), BH))  # Christmas Day, a Friday

    def test_working_days_in_range(self):
        self.assertEqual(len(working_days(date(2026, 11, 23), date(2026, 11, 25), BH)), 3)
        # 21 Dec 2026 to 1 Jan 2027: 25 and 28 Dec and 1 Jan are bank holidays, plus two weekends
        span = working_days(date(2026, 12, 21), date(2027, 1, 1), BH)
        self.assertEqual([d.isoformat() for d in span], ["2026-12-21", "2026-12-22", "2026-12-23", "2026-12-24",
                                                         "2026-12-29", "2026-12-30", "2026-12-31"])

    def test_single_day_and_weekend_only_ranges(self):
        self.assertEqual(working_days(date(2026, 11, 23), date(2026, 11, 23), BH), [date(2026, 11, 23)])
        self.assertEqual(working_days(date(2026, 11, 21), date(2026, 11, 22), BH), [])

    def test_end_before_start(self):
        with self.assertRaises(ValueError):
            working_days(date(2026, 11, 25), date(2026, 11, 23), BH)

    def test_non_working_days(self):
        self.assertEqual(non_working_days(date(2026, 12, 24), date(2026, 12, 29), BH),
                         [date(2026, 12, 25), date(2026, 12, 26), date(2026, 12, 27), date(2026, 12, 28)])


class ShiftTests(unittest.TestCase):
    def test_forward_and_back_are_inverse_on_working_days(self):
        self.assertEqual(shift_working_days(date(2026, 11, 16), -10, BH), date(2026, 11, 2))
        self.assertEqual(shift_working_days(date(2026, 11, 2), 10, BH), date(2026, 11, 16))

    def test_skips_bank_holidays(self):
        # back from Fri 8 Jan 2027: 7, 6, 5, 4 Jan; skip 1 Jan; 31, 30, 29 Dec; skip 28, 25; 24, 23, 22 Dec
        self.assertEqual(shift_working_days(date(2027, 1, 8), -10, BH), date(2026, 12, 22))

    def test_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            shift_working_days(date(2026, 11, 2), 0, BH)


class ParseTests(unittest.TestCase):
    def test_uk_day_first_and_iso_are_accepted(self):
        for text in ("23/11/2026", " 23/11/2026 ", "2026-11-23"):
            self.assertEqual(parse_date(text), date(2026, 11, 23))
        self.assertEqual(parse_date("03/11/2026"), date(2026, 11, 3))  # 3 November: day first, never US style
        self.assertEqual(parse_date("3/11/2026"), date(2026, 11, 3))

    def test_format_is_checked_before_interpreting(self):
        # Python's own date.fromisoformat accepts the first two; the format check refuses them.
        for bad in ("20261123", "2026-W48-1", "23/11/26", "23.11.2026", "23-11-2026", "2026/11/23",
                    "23 November 2026", "next Friday", "", None, 20261123):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as cm:
                parse_date(bad)
            self.assertIn("DD/MM/YYYY", str(cm.exception))

    def test_impossible_dates_are_refused(self):
        for bad in ("31/02/2026", "2026-02-31", "00/11/2026", "23/13/2026"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_date(bad)

    def test_earlier_name_still_works(self):
        self.assertIs(parse_iso_date, parse_date)

if __name__ == "__main__":
    unittest.main()
