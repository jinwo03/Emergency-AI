"""증상 분석 (C 담당).

흐름
1. 너무 짧은 첫 발화는 LLM 호출 없이 바로 되묻는다.
2. Local LLM(Ollama)에 JSON 스키마를 강제해 KTAS·진료과·E-Gen 질환 코드를 받는다.
3. 결과를 검증·정규화한다 (허용 목록 밖 값 제거, 진료과 표기 통일).
4. 위험 신호 키워드가 있으면 KTAS 상한을 강제한다 (Under-triage 방지 안전장치).
5. LLM을 쓸 수 없으면 규칙 기반 결과로 대체한다.

환경 변수
- TRIAGE_BACKEND : "llm"(기본) 또는 "rule" (테스트·LLM 없는 PC용)
- TRIAGE_MODEL   : Ollama 모델 이름 (기본 qwen2.5:7b)
- OLLAMA_URL     : Ollama 주소 (기본 http://localhost:11434)
"""
import json
import logging
import os
import re

import requests

from algorithm.egen_codes import DEPARTMENTS, EGEN_CONDITIONS, normalize_dept, valid_condition
from contracts.schemas import TriageResult, TriageStatus

log = logging.getLogger(__name__)

MIN_TEXT_LEN = 10        # 이보다 짧은 첫 발화는 바로 되묻기
MAX_FOLLOW_UPS = 2       # 되묻기 최대 횟수 (응급 상황이라 길게 끌지 않음)
LLM_TIMEOUT_SEC = 30

DEFAULT_FOLLOW_UP = "어떤 증상이 있으신지 조금 더 자세히 말씀해 주세요."

# ── 위험 신호 → KTAS 상한 ─────────────────────────────────────────────
# LLM이 덜 위급하게 판단해도 이 표현이 있으면 KTAS를 이 값 이하로 강제한다
RED_FLAGS: list[tuple[str, int]] = [
    (r"의식(이|을)?\s*(없|잃)|반응(이)?\s*없|숨(을)?\s*(안|못)\s*쉬|호흡(이)?\s*없|심정지|맥(이)?\s*없", 1),
    (r"가슴.{0,6}(아프|아파|통증|조이|조여|쥐어짜|답답)|흉통", 2),
    (r"말(이)?\s*어눌|발음(이)?\s*(이상|어눌)|(팔|다리)(에|가)?\s*힘(이)?\s*(없|빠)|한쪽.{0,4}(마비|저림|처짐)|편마비|얼굴.{0,4}(처지|비뚤)", 2),
    (r"피를\s*토|토혈|각혈|객혈|피를\s*기침", 2),
    (r"경련|발작", 2),
    (r"숨(이)?\s*(차|가빠|막히)|호흡\s*곤란|숨쉬기\s*(힘들|어렵)", 2),
    (r"머리(가)?\s*깨질|벼락\s*두통|극심한\s*두통", 2),
]


def _ktas_ceiling(text: str) -> int | None:
    """위험 신호가 있으면 허용되는 최대 KTAS 값을, 없으면 None을 돌려준다"""
    ceilings = [level for pattern, level in RED_FLAGS if re.search(pattern, text)]
    return min(ceilings) if ceilings else None


# ── LLM 프롬프트·스키마 ──────────────────────────────────────────────
_CONDITION_LINES = "\n".join(
    f"- {code}: {info['name']} (예: {', '.join(info['keywords'][:3])})"
    for code, info in EGEN_CONDITIONS.items()
)

SYSTEM_PROMPT = f"""너는 한국 응급실 이송 전 단계의 중증도 분류 보조 도구다.
보호자의 말을 읽고 아래 JSON 형식으로만 답한다. 설명 문장은 쓰지 않는다.

[KTAS 중증도: 숫자가 작을수록 위급]
1 = 소생: 의식 없음, 호흡·맥박 없음, 심정지
2 = 긴급: 흉통·호흡곤란, 갑작스런 한쪽 마비나 말 어눌함, 토혈, 경련, 극심한 두통
3 = 응급: 지속되는 복통, 고열, 골절 의심, 출혈은 있으나 의식 명료
4 = 준응급: 가벼운 열상, 염좌, 경미한 통증
5 = 비응급: 감기 증상, 오래된 경미한 증상
판단이 애매하면 더 위급한 쪽(작은 숫자)을 고른다.

[의심 질환 코드: 해당될 때만 최대 3개, 없으면 빈 배열]
{_CONDITION_LINES}

[진료과] 반드시 허용 목록 안에서 최대 3개를 고른다.

[되묻기]
- 무슨 증상인지 알 수 없을 때만 need_more_info=true 로 하고, 짧은 존댓말 질문 하나를 follow_up_question 에 쓴다.
- 위험 신호(의식 저하, 호흡 이상, 흉통, 마비 등)가 이미 있으면 되묻지 말고 바로 분류한다.
- need_more_info=false 이면 follow_up_question 은 빈 문자열로 둔다."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "need_more_info": {"type": "boolean"},
        "follow_up_question": {"type": "string"},
        "ktas": {"type": "integer", "minimum": 1, "maximum": 5},
        "departments": {"type": "array", "items": {"type": "string", "enum": DEPARTMENTS}},
        "conditions": {"type": "array", "items": {"type": "string", "enum": list(EGEN_CONDITIONS)}},
    },
    "required": ["need_more_info", "follow_up_question", "ktas", "departments", "conditions"],
}


def _build_user_message(text: str, age: int | None, history: list[str]) -> str:
    """이전 발화까지 합쳐 LLM에 넘길 사용자 메시지를 만든다"""
    lines = []
    if age is not None:
        lines.append(f"환자 나이: {age}세")
    for i, prev in enumerate(history, 1):
        lines.append(f"이전 발화 {i}: {prev}")
    lines.append(f"이번 발화: {text}")
    if len(history) >= MAX_FOLLOW_UPS:
        lines.append("※ 더 이상 되묻지 말고 지금 정보로 분류할 것")
    return "\n".join(lines)


def _call_llm(text: str, age: int | None, history: list[str]) -> dict:
    """Ollama 호출 후 JSON(dict)을 돌려준다. 실패하면 예외를 던진다"""
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
    payload = {
        "model": os.environ.get("TRIAGE_MODEL", "qwen2.5:7b"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(text, age, history)},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT_SEC)
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


# ── 규칙 기반 대체 경로 ──────────────────────────────────────────────
def _rule_based(text: str) -> dict:
    """LLM을 쓸 수 없을 때의 최소 규칙. 기본은 응급의학과, 위험 신호만 반영"""
    departments, conditions = ["응급의학과"], []
    if re.search(RED_FLAGS[1][0], text):
        departments, conditions = ["순환기내과"], ["Y0010"]
    elif re.search(RED_FLAGS[2][0], text):
        departments, conditions = ["신경과"], ["Y0020"]
    return {"need_more_info": False, "follow_up_question": "", "ktas": 3,
            "departments": departments, "conditions": conditions}


# ── 결과 검증·정규화 ─────────────────────────────────────────────────
def _to_result(raw: dict, full_text: str, history: list[str]) -> TriageResult:
    """LLM/규칙 결과를 계약 형식으로 바꾸고 안전장치를 적용한다"""
    can_ask = len(history) < MAX_FOLLOW_UPS
    if raw.get("need_more_info") and can_ask and _ktas_ceiling(full_text) is None:
        question = (raw.get("follow_up_question") or "").strip() or DEFAULT_FOLLOW_UP
        return TriageResult(status=TriageStatus.need_more_info, follow_up_question=question)

    # 진료과: 표기 통일 + 허용 목록 밖 값 제거 + 중복 제거
    departments = []
    for d in raw.get("departments") or []:
        nd = normalize_dept(str(d).strip())
        if nd and nd not in departments:
            departments.append(nd)
    if not departments:
        departments = ["응급의학과"]

    # 질환 코드: E-Gen 27개 코드만 남김
    conditions = []
    for c in raw.get("conditions") or []:
        code = str(c).strip().upper()
        if valid_condition(code) and code not in conditions:
            conditions.append(code)

    # KTAS: 범위 보정 후 위험 신호 상한 적용
    try:
        ktas = int(raw.get("ktas"))
    except (TypeError, ValueError):
        ktas = 3
    ktas = min(max(ktas, 1), 5)
    ceiling = _ktas_ceiling(full_text)
    if ceiling is not None:
        ktas = min(ktas, ceiling)

    return TriageResult(status=TriageStatus.complete, ktas=ktas,
                        departments=departments[:3], conditions=conditions[:3])


def analyze_symptoms(text: str, age: int | None, history: list[str]) -> TriageResult:
    """보호자 발화에서 중증도/필요 진료과/의심 질환 코드를 추출한다.

    Args:
        text: 이번 발화 (STT 결과)
        age: 환자 나이, 모르면 None
        history: 같은 세션의 이전 발화들 (오래된 순). 첫 요청이면 빈 리스트.

    Returns:
        정보가 충분하면 status=complete + ktas/departments/conditions(E-Gen 코드),
        부족하면 status=need_more_info + follow_up_question.
    """
    full_text = " ".join(history + [text])

    # 1. 너무 짧은 첫 발화는 위험 신호가 없으면 바로 되묻기
    if len(text) < MIN_TEXT_LEN and not history and _ktas_ceiling(text) is None:
        return TriageResult(status=TriageStatus.need_more_info, follow_up_question=DEFAULT_FOLLOW_UP)

    # 2. LLM 호출 (실패 시 규칙 기반으로 대체)
    raw = None
    if os.environ.get("TRIAGE_BACKEND", "llm") == "llm":
        try:
            raw = _call_llm(text, age, history)
        except Exception as exc:  # 연결 실패, 시간 초과, JSON 오류 등
            log.warning("LLM triage 실패, 규칙 기반으로 대체: %s", exc)
    if raw is None:
        raw = _rule_based(full_text)

    # 3. 검증·정규화·안전장치
    return _to_result(raw, full_text, history)
