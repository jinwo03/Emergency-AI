"""응급 메시지 원문에서 시간 조건을 규칙으로 뽑는다 (분류 모델은 시간을 학습하지 않는다).

두 가지 꼴만 다룬다. 확실하지 않으면 None 을 돌려주고, 그러면 시간 조건 없이 '항상'으로 둔다.
- 제한 꼴: "평일 야간 및 주말 진료 불가" → 그 시간에 제한 → [(시간, 상태)]
- 허용 꼴: "평일 8시~17시만 가능", "정규시간 외 불가" → 그 시간은 가능, 그 외는 제한 → [(시간, 가능), (그 외, 상태)]

결과 시간 표기는 time_rule 형식이며, 만든 뒤 time_rule.parse 로 검사한다.
"""
import re

from preprocessing import time_rule

DAY_NAMES = "월화수목금토일"
_HOUR = r"(\d{1,2})(?:[:시]\s*(\d{2})?\s*분?)?"
_AMPM = re.compile(r"(\d{1,2})\s*(AM|PM)\s*[~\-]\s*(?:익일\s*)?(\d{1,2})\s*(AM|PM)", re.I)
_RANGE = re.compile(rf"(?<![\d./])(?:오전|오후|저녁|아침)?\s*{_HOUR}\s*[~\-]\s*(?:익일|다음\s*날)?\s*(?:오전|오후|저녁|아침)?\s*{_HOUR}(?![\d./])")
_AFTER = re.compile(rf"(저녁|오후)?\s*(\d{{1,2}})(?::(\d{{2}}))?\s*시?\s*(?:이후|부터)(?!\s*\d)")
_UNTIL = re.compile(r"(?:매일\s*)?~\s*(\d{1,2}):(\d{2})\s*(?:까지)?\s*가능")
_REGULAR = re.compile(r"정규\s*(?:진료\s*)?시간|일과\s*시간|외래\s*(?:진료\s*)?(?:가능\s*)?시간|상시\s*시간")
_DAY_LIST = re.compile(rf"(?<![가-힣])([{DAY_NAMES}])(?:요일)?(?:\s*[,/·.]\s*([{DAY_NAMES}])(?:요일)?)+")
_DAY_SPAN = re.compile(rf"(?<![가-힣])([{DAY_NAMES}])(?:요일)?\s*[~\-]\s*([{DAY_NAMES}])(?:요일)?")
_ONE_DAY = re.compile(rf"(?:매주\s*)?([{DAY_NAMES}])요일")
_HOLIDAY_WORDS = ["주말", "공휴일", "휴일"]
_LIMIT = re.compile(r"불가|제한|어렵|어려|부재|원활하지|중단")
_ALLOW_ANY = re.compile(r"가능\s*(?:시간|일시)|(?:수용|진료)\s*가능|만\s*(?:진료\s*)?가능|까지만|시간만|이외|외\s*시간|"
                        r"그\s*외|진료\s*시간|가능합니다|가능\s*\(")
_EXISTING = re.compile(r"f/?u|기존|재진|본원\s*(?:치료|환자)", re.I)
_DAYS_BEFORE = re.compile(rf"(평일|주말|공휴일|휴일|(?<![가-힣])[{DAY_NAMES}](?:요일)?(?:\s*[~\-,·/]\s*[{DAY_NAMES}](?:요일)?)*)\s*\(?\s*$")


def _hm(h: str, m: str | None, pm: bool = False) -> tuple[int, int] | None:
    hour, minute = int(h), int(m or 0)
    if pm and hour < 12:
        hour += 12
    if hour == 24:
        hour = 0
    return (hour, minute) if 0 <= hour < 24 and 0 <= minute < 60 else None


def _fmt(a: tuple[int, int], b: tuple[int, int]) -> str:
    return f"{a[0]:02d}:{a[1]:02d}-{b[0]:02d}:{b[1]:02d}"


def find_clocks(text: str) -> list[tuple[str, int, int]]:
    """숫자 시각 범위들 [(시간 표기, 시작, 끝)]. 없으면 '~시 이후' → '~까지 가능' → 주간/야간 중 하나."""
    found = []
    for m in _AMPM.finditer(text):
        a, b = _hm(m[1], None, m[2].upper() == "PM"), _hm(m[3], None, m[4].upper() == "PM")
        if a and b:
            found.append((_fmt(a, b), m.start(), m.end()))
    for m in _RANGE.finditer(text):
        if any(s <= m.start() < e for _, s, e in found):
            continue
        before = text[max(0, m.start() - 3):m.start()]
        if re.search(r"\d\s*[./]\s*$", before):       # 날짜(10.30 ~ 10.31)
            continue
        evening = re.search(r"(저녁|오후)\s*$", text[max(0, m.start() - 4):m.start() + 3])
        a, b = _hm(m[1], m[2], bool(evening)), _hm(m[3], m[4])
        if a and b and a != b and ("시" in m[0] or ":" in m[0]):
            if b < a and b[0] < 12 and "익일" not in m[0] and a[0] < 12 and b[0] + 12 < 24 and (b[0] + 12, b[1]) > a:
                b = (b[0] + 12, b[1])                  # 9~5 → 09:00-17:00
            found.append((_fmt(a, b), m.start(), m.end()))
    if found:
        return sorted(found, key=lambda x: x[1])
    one = find_clock(text)
    return [one] if one else []


def find_clock(text: str) -> tuple[str, int, int] | None:
    if m := _AFTER.search(text):
        a = _hm(m[2], m[3], bool(m[1]))
        if a and a[0] >= 12:
            return _fmt(a, (8, 0)), m.start(), m.end()
    if m := _UNTIL.search(text):
        b = _hm(m[1], m[2])
        if b:
            return _fmt(b, (8, 0)), m.start(), m.end()
    for word, spec in (("야간", "야간"), ("주간", "주간"), ("저녁", "야간")):
        i = text.find(word)
        if i >= 0:
            return spec, i, i + len(word)
    return None


def find_days(text: str) -> str | None:
    if "평일" in text:
        return "평일"
    if m := _DAY_SPAN.search(text):
        return f"{m[1]}~{m[2]}"
    if m := _DAY_LIST.search(text):
        days = re.findall(rf"[{DAY_NAMES}]", m[0].replace("요일", ""))
        return "·".join(dict.fromkeys(days))
    if m := _ONE_DAY.search(text):
        return m[1]
    return None


def _holidays(text: str, days: str | None) -> list[str]:
    found = [w for w in _HOLIDAY_WORDS if w in text]
    if "공휴일" in found and "휴일" in found:
        found.remove("휴일")
    if "주말" in found and "공휴일" in found:
        found = ["휴일"]
    if days and days in ("토·일", "토~일"):
        return []
    return found


def _is_allow(text: str, start: int, end: int, regular: bool) -> bool:
    """허용 꼴인지. 시각 바로 뒤에 제한 단어가 붙으면 제한 꼴, 기존 환자만 문구는 제한 꼴로 본다."""
    if regular and re.match(r"^\W{0,3}(외|이외|이후)", text[end:end + 6]):
        return True
    if _EXISTING.search(text):
        return False
    near = text[end:end + 10]
    limit, allow = _LIMIT.search(near), re.search("가능", near)
    if limit and (not allow or limit.start() < allow.start()):
        return False
    return bool(_ALLOW_ANY.search(text))


def extract(text: str, status: str) -> list[tuple[str, str]] | None:
    """원문과 (분류 모델이 고른) 상태로 [(시간, 상태), ...]. 시간 조건을 못 찾으면 None."""
    if status not in ("불가", "어려움", "기존 환자만"):
        return None
    if "24시간" in text and not re.search(r"\d{1,2}\s*[시:]", text.replace("24시간", "")):
        return None
    regular = _REGULAR.search(text)
    clocks = find_clocks(text)
    clock = clocks[0] if clocks else None
    if regular and clock and clock[0] in ("야간", "주간"):
        clock = None                                   # '정규시간만 가능 (야간 불가)' → 정규시간 기준
    days = find_days(text)
    if regular and not clock:
        spec, start, end = "평일 주간", regular.start(), regular.end()
        allow = _is_allow(text, start, end, True) or "외" in text[end:end + 4] or "가능" in text[end:end + 10]
    elif clock:
        items, prev = [], 0
        for k, (spec_clock, s, e) in enumerate(clocks):
            d = None
            if m := _DAYS_BEFORE.search(text[max(prev, s - 12):s]):
                d = find_days(m[1]) or m[1]
            elif k == 0:
                d = days or ("평일" if regular else None)
            items.append(f"{d} {spec_clock}" if d else spec_clock)
            prev = e
        spec = ", ".join(dict.fromkeys(items))
        start, end = clock[1], clock[2]
        days = days or None
        allow = _is_allow(text, start, end, bool(regular))
    elif days:
        i = text.find(days[0])
        spec, start, end, allow = days, i, i + 1, False
    else:
        hol = _holidays(text, None)
        if not hol:
            return None
        spec, allow = ", ".join(hol), False
    if not allow:
        extra = [h for h in _holidays(text, days) if h not in spec]
        spec = ", ".join([spec, *extra])
    try:
        time_rule.parse(spec)
    except ValueError:
        return None
    return [(spec, "가능"), (time_rule.OTHERWISE, status)] if allow else [(spec, status)]
