from datetime import date, datetime

import pytest

from preprocessing import time_rule

# 2026-11-02 은 월요일
MON, TUE, FRI, SAT, SUN = (datetime(2026, 11, d) for d in (2, 3, 6, 7, 8))


def at(day: datetime, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return day.replace(hour=h, minute=m)


@pytest.mark.parametrize("spec, when, expected", [
    ("항상", at(SUN, "03:00"), True),
    ("평일 08:00-17:00", at(MON, "09:00"), True),
    ("평일 08:00-17:00", at(MON, "17:00"), False),
    ("평일 08:00-17:00", at(SAT, "09:00"), False),
    ("평일 21:00-08:00", at(SAT, "07:00"), True),     # 금요일 밤에 시작한 시간
    ("평일 21:00-08:00", at(MON, "07:00"), False),    # 일요일 밤은 평일이 아님
    ("야간", at(TUE, "02:00"), True),
    ("평일 야간, 휴일", at(SUN, "12:00"), True),
    ("월~목 08:00-16:00", at(FRI, "09:00"), False),
    ("금~일", at(SUN, "23:00"), True),
    ("월·수·금 08:00-08:00", at(TUE, "07:59"), True),  # 24시간
    ("토 13:00-24:00, 일, 월 00:00-08:00", at(MON, "07:30"), True),
    ("주간, 화·목 18:00-22:00", at(TUE, "20:00"), True),
    ("주간, 화·목 18:00-22:00", at(MON, "20:00"), False),
    ("22:00-08:00", at(MON, "23:00"), True),
])
def test_matches(spec, when, expected):
    assert time_rule.matches(spec, when) is expected


def test_holiday_is_not_weekday():
    holiday = {date(2026, 11, 3)}
    assert not time_rule.matches("평일 주간", at(TUE, "10:00"), holiday)
    assert time_rule.matches("공휴일", at(TUE, "10:00"), holiday)
    assert time_rule.matches("휴일", at(TUE, "10:00"), holiday)


@pytest.mark.parametrize("spec", ["평일 25:00-08:00", "평일 8시", "월~", "", "평일 08:60-09:00"])
def test_invalid_spec_raises(spec):
    with pytest.raises(ValueError):
        time_rule.parse(spec)


def test_status_uses_first_match_then_otherwise_then_available():
    rules = [("목", "불가", ""), ("평일 주간", "가능", "지연"), ("그 외", "기존 환자만", "문의")]
    assert time_rule.status_at(rules, at(datetime(2026, 11, 5), "10:00")) == ("불가", "")  # 목요일
    assert time_rule.status_at(rules, at(MON, "10:00")) == ("가능", "지연")
    assert time_rule.status_at(rules, at(MON, "22:00")) == ("기존 환자만", "문의")
    assert time_rule.status_at([("야간", "불가", "")], at(MON, "10:00")) == ("가능", "")


def test_tue_wed_night_exception_before_general_night_rule():
    rules = [("화·수 야간", "가능", ""), ("휴일, 야간", "불가", "")]
    assert time_rule.status_at(rules, at(TUE, "02:00"))[0] == "불가"                       # 월요일 밤
    assert time_rule.status_at(rules, at(datetime(2026, 11, 5), "02:00"))[0] == "가능"     # 수요일 밤
    assert time_rule.status_at(rules, at(SAT, "14:00"))[0] == "불가"
