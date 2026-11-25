"""메시지 정답의 시간 표기를 해석하고, 주어진 시각의 상태를 계산한다.

표기 규칙 (정답 시트와 같다)
- 요일: 월 화 수 목 금 토 일, 평일(월~금, 공휴일 제외), 주말(토·일), 공휴일, 휴일(주말+공휴일), 매일.
  여러 요일은 '·'로 잇고, '월~목'처럼 범위로 쓸 수 있다 ('금~일'처럼 주를 넘어가도 된다).
- 시각: 24시간 'HH:MM-HH:MM'. 끝이 시작보다 이르거나 같으면 다음 날까지이고, 그 시간은 시작한 날의 요일로 본다.
  시각 대신 '주간'(08:00-18:00), '야간'(18:00-08:00)을 쓸 수 있다.
- 요일만 쓰면 그날 하루 전체, 시각만 쓰면 매일이다. 여러 개는 ', '로 잇는다.
- '항상'은 모든 시각, '그 외'는 같은 대상의 다른 줄에 적힌 시간을 뺀 나머지다.

상태 계산: 같은 대상의 줄을 위에서부터 보고 처음 맞는 줄을 쓴다. 맞는 줄이 없으면 '그 외' 줄,
그것도 없으면 메시지가 말하지 않은 시간이므로 '가능'으로 본다.
"""
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

ALWAYS = "항상"
OTHERWISE = "그 외"
DEFAULT_STATUS = "가능"
DAY_START, DAY_END = 8 * 60, 18 * 60

WEEKDAYS = "월화수목금토일"
NAMED_DAYS = {"평일", "주말", "공휴일", "휴일", "매일"}
_TIME = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")
_RANGE = re.compile(rf"^([{WEEKDAYS}])~([{WEEKDAYS}])$")


@dataclass(frozen=True)
class Window:
    days: frozenset[str]   # 요일 글자 또는 평일/주말/공휴일/휴일/매일
    start: int | None      # 분 단위. None 이면 하루 전체
    end: int | None


def _minutes(h: str, m: str) -> int:
    value = int(h) * 60 + int(m)
    if not 0 <= value <= 24 * 60 or int(m) >= 60:
        raise ValueError(f"잘못된 시각: {h}:{m}")
    return value


def _days(token: str) -> set[str]:
    days = set()
    for part in token.split("·"):
        if part in NAMED_DAYS or (len(part) == 1 and part in WEEKDAYS):
            days.add(part)
        elif m := _RANGE.match(part):
            i, j = WEEKDAYS.index(m[1]), WEEKDAYS.index(m[2])
            days.update(WEEKDAYS[(i + k) % 7] for k in range((j - i) % 7 + 1))
        else:
            raise ValueError(f"알 수 없는 요일: {part}")
    return days


def parse(spec: str) -> list[Window] | str:
    """시간 표기를 Window 목록으로 바꾼다. '항상'과 '그 외'는 그대로 돌려준다."""
    spec = spec.strip()
    if spec in (ALWAYS, OTHERWISE):
        return spec
    windows = []
    for item in spec.split(","):
        days, start, end = set(), None, None
        tokens = item.split()
        if not tokens:
            raise ValueError(f"빈 항목: {spec!r}")
        for token in tokens:
            if m := _TIME.match(token):
                start, end = _minutes(m[1], m[2]), _minutes(m[3], m[4])
            elif token == "주간":
                start, end = DAY_START, DAY_END
            elif token == "야간":
                start, end = DAY_END, DAY_START
            else:
                days |= _days(token)
        windows.append(Window(frozenset(days or {"매일"}), start, end))
    return windows


def _day_matches(days: frozenset[str], d: date, holidays: set[date]) -> bool:
    holiday = d in holidays
    weekday = WEEKDAYS[d.weekday()]
    return any(
        x == "매일"
        or x == weekday
        or (x == "평일" and d.weekday() < 5 and not holiday)
        or (x == "주말" and d.weekday() >= 5)
        or (x == "공휴일" and holiday)
        or (x == "휴일" and (d.weekday() >= 5 or holiday))
        for x in days
    )


def matches(spec: str, at: datetime, holidays: set[date] = frozenset()) -> bool:
    """'그 외'는 다른 줄과 함께 봐야 하므로 여기서는 False 로 둔다."""
    windows = parse(spec)
    if windows == ALWAYS:
        return True
    if windows == OTHERWISE:
        return False
    today, minute = at.date(), at.hour * 60 + at.minute
    for w in windows:
        if w.start is None:
            if _day_matches(w.days, today, holidays):
                return True
        elif w.start < w.end:
            if w.start <= minute < w.end and _day_matches(w.days, today, holidays):
                return True
        else:  # 다음 날까지 이어지는 시간. 같으면 24시간
            if minute >= w.start and _day_matches(w.days, today, holidays):
                return True
            if minute < w.end and _day_matches(w.days, today - timedelta(days=1), holidays):
                return True
    return False


def status_at(rules: list[tuple[str, str, str]], at: datetime,
              holidays: set[date] = frozenset()) -> tuple[str, str]:
    """한 대상의 (시간, 상태, 주의) 줄 목록에서 주어진 시각의 (상태, 주의)를 구한다.

    주의는 '', '문의'(못 받을 수도 있어 확인 필요), '지연'(받지만 늦어짐), '문의·지연' 중 하나다.
    """
    otherwise = None
    for spec, status, caution in rules:
        if spec.strip() == OTHERWISE:
            otherwise = otherwise or (status, caution)
        elif matches(spec, at, holidays):
            return status, caution
    return otherwise or (DEFAULT_STATUS, "")
