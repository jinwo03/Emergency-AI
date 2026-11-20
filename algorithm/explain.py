"""추천 사유 문장 생성 (C 담당).

병원마다 '왜 이 순위인지'를 보호자·구급대원이 바로 이해할 수 있는 한국어 한두 문장으로 만든다.
순위는 절대 바꾸지 않는다. 문장만 만든다.

입력
- 병원 정보 (RankedHospital): 도착 시간, 병상, 정보 갱신 시각, 수용 곤란 메시지, 수용 확률
- 환자 정보: KTAS, 필요 진료과, 의심 질환 코드
- contributions (선택): B 모델의 SHAP 기여도 {피처 이름: 값}. 양수=수용 확률을 높임, 음수=낮춤

LLM 다듬기 (선택, 환경 변수 EXPLAIN_LLM=1)
- 템플릿 문장을 Local LLM이 자연스럽게 고쳐 쓴다.
- 고친 문장에 원래 숫자가 하나라도 빠지면 버리고 템플릿 문장을 쓴다 (숫자 왜곡 방지).
"""
import logging
import os
import re

import requests

from algorithm.egen_codes import EGEN_CONDITIONS
from contracts.schemas import RankedHospital

log = logging.getLogger(__name__)

# B 모델 피처 이름 → 사람이 읽는 이름. B와 피처 이름 확정되면 여기만 맞추면 됨
FEATURE_LABELS = {
    "beds_available": "응급실 가용 병상", "hvec": "응급실 가용 병상",
    "beds_trend": "최근 병상 변화", "hvec_delta": "최근 병상 변화",
    "data_age_min": "정보 갱신 시각", "staleness": "정보 갱신 시각",
    "eta_min": "도착 시간", "distance_km": "거리",
    "hour": "시간대", "dow": "요일", "weekday": "요일", "is_holiday": "공휴일 여부",
    "has_block_msg": "수용 곤란 메시지", "msg_block": "수용 곤란 메시지",
    "dept_available": "필요 진료과 진료 가능 여부", "sym_available": "해당 질환 수용 가능 여부",
    "accept_rate_7d": "최근 7일 수용 가능 비율", "accept_rate_24h": "최근 24시간 수용 가능 비율",
}

STALE_MIN = 60     # 이보다 오래된 병상 정보는 주의 문구
FRESH_MIN = 10


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _beds_text(beds: int | None) -> str:
    if beds is None:
        return "병상 정보 없음"
    if beds > 0:
        return f"가용 병상 {beds}개"
    return "가용 병상 없음(과밀)"


def _relevant_notices(notices: list[str], departments: list[str], conditions: list[str]) -> list[str]:
    """환자에게 필요한 질환·진료과와 관련된 수용 곤란 메시지만 고른다"""
    names = [EGEN_CONDITIONS[c]["name"] for c in conditions if c in EGEN_CONDITIONS]
    keys = [n.split("(")[0] for n in names] + list(departments)
    return [n for n in notices if any(k and k in n for k in keys)]


def explain_template(
    hospital: RankedHospital,
    ktas: int,
    departments: list[str],
    conditions: list[str],
    contributions: dict[str, float] | None = None,
) -> str:
    """사실과 SHAP 기여도로 추천 사유 문장을 만든다 (결정적, LLM 없음)"""
    parts = [f"예상 도착 {hospital.eta_min}분({hospital.distance_km:.1f}km), {_beds_text(hospital.beds_available)}"]

    if hospital.accept_prob is not None:
        parts[0] += f", 도착 시점 수용 확률 {round(hospital.accept_prob * 100)}%"

    if contributions:
        ups = sorted(((v, k) for k, v in contributions.items() if v > 0), reverse=True)[:2]
        downs = sorted((v, k) for k, v in contributions.items() if v < 0)[:1]
        if ups:
            parts.append("확률을 높인 요인: " + ", ".join(_label(k) for _, k in ups))
        if downs:
            parts.append("낮춘 요인: " + ", ".join(_label(k) for _, k in downs))

    age = hospital.data_age_min
    if age is not None and age >= STALE_MIN:
        parts.append(f"병상 정보가 {age}분 전 것이라 실제와 다를 수 있음")
    elif age is not None and age <= FRESH_MIN:
        parts.append("병상 정보 최근 갱신")

    warn = _relevant_notices(hospital.notices, departments, conditions)
    if warn:
        parts.append(f"주의: '{warn[0]}' 메시지가 있음")

    sentence = ". ".join(parts) + "."
    if ktas <= 2 or warn:
        sentence += " 출발 전 전화로 수용 가능 여부를 꼭 확인하세요."
    return sentence


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _polish_with_llm(sentence: str) -> str:
    """템플릿 문장을 LLM으로 다듬는다. 숫자가 바뀌거나 빠지면 원문을 돌려준다"""
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
    payload = {
        "model": os.environ.get("TRIAGE_MODEL", "qwen2.5:7b"),
        "messages": [
            {"role": "system", "content": "다음 응급실 추천 사유를 보호자가 이해하기 쉬운 존댓말 한두 문장으로 고쳐 써라. "
                                          "숫자와 병원 정보는 절대 바꾸거나 빼지 말고, 새로운 사실을 추가하지 마라. 문장만 출력."},
            {"role": "user", "content": sentence},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        out = resp.json()["message"]["content"].strip()
    except Exception as exc:
        log.warning("explain LLM 실패, 템플릿 사용: %s", exc)
        return sentence
    if not out or sorted(_numbers(out)) != sorted(_numbers(sentence)):
        return sentence
    return out


def explain(
    hospital: RankedHospital,
    ktas: int,
    departments: list[str],
    conditions: list[str],
    contributions: dict[str, float] | None = None,
) -> str:
    """병원 추천 사유 문장. 순위는 바꾸지 않는다."""
    sentence = explain_template(hospital, ktas, departments, conditions, contributions)
    if os.environ.get("EXPLAIN_LLM") == "1":
        return _polish_with_llm(sentence)
    return sentence
