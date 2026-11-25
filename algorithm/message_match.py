"""병원 메시지 줄(정답 시트 형식)과 환자 정보로 주어진 시각의 수용 상태를 구한다.

줄: [대상, 진료과, 질환, 조건, 시간, 상태, 주의]. 진료과·질환·조건은 ", " 로 이은 값이다.

맞추는 법 (정답 시트 안내와 같다)
- 진료과: 줄의 진료과 중 하나가 환자 진료과에 있거나 줄이 '전체'면 맞다.
- 질환: 비었거나 줄의 코드 중 하나가 환자 질환 코드에 있으면 맞다.
- 조건: 비었거나 줄의 조건을 환자가 모두 가지면 맞다.
- 환자에게 맞는 줄을 위에서부터 보고, 시간이 처음 맞는 줄을 쓴다. 어느 줄도 맞지 않으면 '가능'.
- 시간 '그 외'는 진료과·질환·조건이 같은 다른 줄의 시간이 모두 맞지 않을 때 맞는다.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache

from contracts.vocab import ALL_DEPARTMENTS
from preprocessing import time_rule

TARGET, DEPT, CODES, CONDS, TIME, STATUS, CAUTION = range(7)
DEFAULT = ("가능", "", None)


@dataclass(frozen=True)
class Patient:
    departments: frozenset[str] = field(default_factory=frozenset)
    disease_codes: frozenset[str] = field(default_factory=frozenset)
    conditions: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, departments=(), disease_codes=(), conditions=()) -> "Patient":
        return cls(frozenset(departments), frozenset(disease_codes), frozenset(conditions))


def values(cell) -> frozenset[str]:
    return frozenset(v.strip() for v in str(cell or "").split(",") if v.strip())


def line_applies(line: list, patient: Patient) -> bool:
    depts, codes, conds = values(line[DEPT]), values(line[CODES]), values(line[CONDS])
    if ALL_DEPARTMENTS not in depts and not depts & patient.departments:
        return False
    if codes and not codes & patient.disease_codes:
        return False
    return conds <= patient.conditions


@lru_cache(maxsize=200_000)
def _time_matches(spec: str, at: datetime, holidays: frozenset[date]) -> bool:
    return time_rule.matches(spec, at, holidays)


def _key(line: list) -> tuple:
    return values(line[DEPT]), values(line[CODES]), values(line[CONDS])


def status_for(lines: list[list], patient: Patient, at: datetime,
               holidays: frozenset[date] = frozenset()) -> tuple[str, str, int | None]:
    """(상태, 주의, 걸린 줄 번호 0부터). 걸린 줄이 없으면 ('가능', '', None)."""
    for i, line in enumerate(lines):
        if not line_applies(line, patient):
            continue
        spec = (line[TIME] or time_rule.ALWAYS).strip()
        if spec == time_rule.OTHERWISE:
            key = _key(line)
            others = [x for j, x in enumerate(lines) if j != i and _key(x) == key
                      and (x[TIME] or "").strip() != time_rule.OTHERWISE]
            hit = not any(_time_matches((x[TIME] or time_rule.ALWAYS).strip(), at, holidays) for x in others)
        else:
            hit = _time_matches(spec, at, holidays)
        if hit:
            return line[STATUS], line[CAUTION] or "", i
    return DEFAULT
