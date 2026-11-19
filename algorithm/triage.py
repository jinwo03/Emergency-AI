"""증상 분석 (C 담당) v2.

흐름
1. 짧고(15자 이하) 증상 단어가 하나도 없는 발화는 LLM 호출 없이 바로 되묻는다.
2. Local LLM(Ollama)에 JSON 스키마를 강제해 KTAS·진료과·E-Gen 질환 코드를 받는다.
3. 증상 규칙(triage_rules)으로 질환 코드를 보충하고 KTAS 상한을 건다 (Under-triage 방지).
4. 나이·문맥과 맞지 않는 질환 코드를 지우고, 진료과를 질환에 맞게 정렬한다.
5. LLM을 쓸 수 없으면 규칙만으로 결과를 만든다.

v1 대비 변경 (eval/v1_run 의 오류 분석 결과 반영)
- 증상이 있는 문장에서 LLM이 되묻는 것을 막음 (위급 환자 지연 7건)
- 질환 코드가 나오면 그 질환의 기본 KTAS 상한 적용 (코드는 맞히고 KTAS만 낮은 경우)
- 증상 규칙 확대: 화상, 절단, 위장관 출혈, 영아 발열, 이물 삼킴 등 (질환·KTAS 모두 놓친 37건)
- 부정·비급성 표현("가슴은 안 아프고", "예전에", "한 달 됐어요")에서는 규칙을 끔
- 성인에게 소아청소년과, 맥락 없는 산과 코드 등 나이·문맥 오류 제거
- 프롬프트에 정답 예시(few-shot) 추가, Ollama 모델 상주(keep_alive)로 시간 초과 방지

환경 변수
- TRIAGE_BACKEND : "llm"(기본) 또는 "rule" (테스트·LLM 없는 PC용)
- TRIAGE_MODEL   : Ollama 모델 이름 (기본 qwen2.5:7b)
- OLLAMA_URL     : Ollama 주소 (기본 http://localhost:11434)
"""
import json
import logging
import os

import requests

from algorithm import triage_rules as rules
from algorithm.egen_codes import DEPARTMENTS, EGEN_CONDITIONS, normalize_dept, valid_condition
from contracts.schemas import TriageResult, TriageStatus

log = logging.getLogger(__name__)

MAX_FOLLOW_UPS = 2       # 되묻기 최대 횟수 (응급 상황이라 길게 끌지 않음)
ASK_MAX_LEN = 15         # 되묻기는 이 길이 이하의 짧은 발화에만 (길게 설명했으면 일단 분류)
LLM_TIMEOUT_SEC = 60     # 첫 호출은 모델을 GPU에 올리느라 오래 걸릴 수 있음
LLM_KEEP_ALIVE = "30m"   # 호출 사이에 모델을 내리지 않음

FOLLOW_UPS = [
    "환자분이 지금 의식이 있고 숨은 잘 쉬나요? 어디가 어떻게 아픈지도 말씀해 주세요.",
    "언제부터 그랬는지, 다치거나 쓰러진 적이 있는지 알려주세요.",
]
DEFAULT_FOLLOW_UP = FOLLOW_UPS[0]

# 평가용: 마지막 호출이 LLM 결과였는지 규칙 대체였는지 기록
LAST_SOURCE = "none"

# ── LLM 프롬프트·스키마 ──────────────────────────────────────────────
_CONDITION_LINES = "\n".join(
    f"- {code}: {info['name']} (예: {', '.join(info['keywords'][:3])})"
    for code, info in EGEN_CONDITIONS.items()
)

FEW_SHOT = """[예시]
말: 할머니가 검은 변을 보시고 어지럽대요 (82세)
답: {"ktas": 2, "departments": ["소화기내과", "내과"], "conditions": ["Y0081"]}
말: 아이가 끓는 물에 팔과 배를 데었어요 (4세)
답: {"ktas": 2, "departments": ["외과", "성형외과", "소아청소년과"], "conditions": ["Y0120"]}
말: 작업하다 손가락이 잘렸어요 (37세)
답: {"ktas": 2, "departments": ["정형외과", "성형외과"], "conditions": ["Y0131"]}
말: 생후 2개월 아기가 열이 나요 (0세)
답: {"ktas": 2, "departments": ["소아청소년과"], "conditions": []}
말: 발목을 삐었는데 걸을 수는 있어요 (27세)
답: {"ktas": 4, "departments": ["정형외과"], "conditions": []}
말: 감기 기운이 있고 콧물이 나요 (26세)
답: {"ktas": 5, "departments": ["내과", "이비인후과"], "conditions": []}"""

SYSTEM_PROMPT = f"""너는 한국 응급실 이송 전 단계의 중증도 분류 보조 도구다.
보호자의 말을 읽고 JSON으로만 답한다. 설명 문장은 쓰지 않는다.

[KTAS 중증도: 숫자가 작을수록 위급]
1 = 소생: 의식 없음, 호흡·맥박 없음, 심정지, 기도 막힘, 대량 출혈
2 = 긴급: 흉통, 호흡곤란, 갑작스런 한쪽 마비·말 어눌함, 토혈·검은 변+어지러움, 경련, 극심한 두통,
    넓은 화상, 손가락·팔다리 절단, 3개월 미만 영아 발열, 아이의 건전지·자석 삼킴, 눈 부상, 자해·타해 위험
3 = 응급: 지속되는 복통, 고열, 골절 의심, 출혈은 있으나 의식 명료
4 = 준응급: 가벼운 열상, 염좌, 경미한 통증
5 = 비응급: 감기 증상, 오래된 경미한 증상, 약 처방
판단이 애매하면 더 위급한 쪽(작은 숫자)을 고른다.

[의심 질환 코드: 해당될 때만 최대 3개, 없으면 빈 배열]
{_CONDITION_LINES}

[진료과] 허용 목록 안에서 최대 3개, 가장 중요한 과를 맨 앞에. 소아청소년과는 18세 이하에만.

{FEW_SHOT}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "ktas": {"type": "integer", "minimum": 1, "maximum": 5},
        "departments": {"type": "array", "items": {"type": "string", "enum": DEPARTMENTS}},
        "conditions": {"type": "array", "items": {"type": "string", "enum": list(EGEN_CONDITIONS)}},
    },
    "required": ["ktas", "departments", "conditions"],
}


def _build_user_message(text: str, age: int | None, history: list[str]) -> str:
    """이전 발화까지 합쳐 LLM에 넘길 사용자 메시지를 만든다"""
    lines = [f"환자 나이: {age}세" if age is not None else "환자 나이: 모름"]
    for i, prev in enumerate(history, 1):
        lines.append(f"이전 발화 {i}: {prev}")
    lines.append(f"이번 발화: {text}")
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
        "keep_alive": LLM_KEEP_ALIVE,
        "options": {"temperature": 0},
    }
    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT_SEC)
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


# ── 결과 조립 ────────────────────────────────────────────────────────
def _merge_conditions(rule_codes: list[str], llm_codes: list, text: str, age: int | None) -> list[str]:
    """규칙 코드를 앞에, LLM 코드를 뒤에 두고 정합성 필터를 거친다"""
    merged: list[str] = []
    for c in list(rule_codes) + [str(c).strip().upper() for c in llm_codes or []]:
        if not valid_condition(c):
            continue
        c = rules.plausible(c, text, age)
        if c and c not in merged:
            merged.append(c)
    return merged[:3]


def _order_departments(llm_depts: list, conditions: list[str], text: str, age: int | None) -> list[str]:
    """첫 질환의 대표 진료과를 맨 앞에 두고, 나이·문맥에 맞지 않는 과를 뺀다"""
    depts: list[str] = []
    if conditions:
        depts.append(EGEN_CONDITIONS[conditions[0]]["depts"][0])
    for d in llm_depts or []:
        nd = normalize_dept(str(d).strip())
        if nd and nd not in depts:
            depts.append(nd)
    if age is not None and age >= 19 and "소아청소년과" in depts:
        depts.remove("소아청소년과")
    if "산부인과" in depts and not (rules.OBSTETRIC.search(text) or rules.GYNECOLOGIC.search(text)):
        depts.remove("산부인과")
    if age is not None and age <= 12 and "소아청소년과" not in depts:
        depts.insert(1 if depts else 0, "소아청소년과")
    return (depts or ["응급의학과"])[:3]


def _final_ktas(llm_ktas, rule_ceiling: int | None, rule_codes: list[str], conditions: list[str], text: str) -> int:
    """LLM KTAS에 규칙 상한과 질환 코드 상한을 적용한다 (낮추는 방향만, 즉 더 위급하게만)"""
    try:
        ktas = min(max(int(llm_ktas), 1), 5)
    except (TypeError, ValueError):
        ktas = 3
    if rule_ceiling is not None:
        ktas = min(ktas, rule_ceiling)
    if not rules.is_non_acute(text):
        for c in conditions:
            if c not in rule_codes:          # 규칙이 직접 찾은 코드는 규칙 상한을 따름
                ktas = min(ktas, rules.CODE_CEILING[c])
    return ktas


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
    global LAST_SOURCE
    full_text = " ".join(history + [text])
    rule_codes, rule_ceiling = rules.scan(full_text, age)

    # 1. 짧고 증상 단어도 위험 신호도 없으면 되묻기 (최대 MAX_FOLLOW_UPS번)
    #    증상 어휘 목록은 완벽할 수 없으므로, 길게 설명한 발화는 되묻지 않고 LLM이 분류한다
    vague = len(text.strip()) <= ASK_MAX_LEN and not rules.has_symptom(full_text)
    if vague and rule_ceiling is None and len(history) < MAX_FOLLOW_UPS:
        LAST_SOURCE = "ask"
        return TriageResult(status=TriageStatus.need_more_info, follow_up_question=FOLLOW_UPS[len(history)])

    # 2. LLM 호출 (실패하면 규칙만 사용)
    raw = None
    if os.environ.get("TRIAGE_BACKEND", "llm") == "llm":
        try:
            raw = _call_llm(text, age, history)
            LAST_SOURCE = "llm"
        except Exception as exc:  # 연결 실패, 시간 초과, JSON 오류 등
            log.warning("LLM triage 실패, 규칙 기반으로 대체: %s", exc)
    if raw is None:
        LAST_SOURCE = "rule"
        raw = {"ktas": 3, "departments": [], "conditions": []}

    # 3. 규칙 보정
    conditions = _merge_conditions(rule_codes, raw.get("conditions"), full_text, age)
    departments = _order_departments(raw.get("departments"), conditions, full_text, age)
    ktas = _final_ktas(raw.get("ktas"), rule_ceiling, rule_codes, conditions, full_text)
    return TriageResult(status=TriageStatus.complete, ktas=ktas, departments=departments, conditions=conditions)
