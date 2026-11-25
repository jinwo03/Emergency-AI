"""전처리 3단계: 메시지를 줄(대상·진료과·질환·조건·시간·상태·주의) 단위로 풀어 5분 격자에 연결한다.

줄 형식과 상태는 정답 시트(label_set)와 같다. 상태는 가능 / 어려움 / 기존 환자만 / 불가 / 판단 불가 중 하나이고,
해석 결과가 없거나 믿을 수 없으면 미해석이다. 미해석은 불가로 취급하면 안 된다.
- 중증 메시지: 질환 코드가 있으므로 코드로 연결하고, 문구로 불가·어려움만 가른다 (code_rule).
- 응급 메시지: 정답 시트의 답(label) → 임시 시험용 답(provisional) → 분류 모델(classifier) 순으로 쓰고, 없으면 문구 규칙만 쓴다 (text_rule).
- 규칙의 '어려움 + 문의'는 '조건에 따라 받을 수도 있으니 확인 필요'라는 뜻이다.

시간 칸은 그대로 둔다. 어느 시각에 어느 줄이 맞는지는 환자 정보와 함께 매칭 단계에서 판단한다.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.grid import SLOT

PARSE_FILE = "message_parse.jsonl"
GENERAL_SYM_CODE = "Y000"
MIN_CONFIDENCE = 0.6

AVAILABLE = "가능"
DIFFICULT = "어려움"
EXISTING_ONLY = "기존 환자만"
BLOCKED = "불가"
UNKNOWN = "판단 불가"
UNPARSED = "미해석"
STATUSES = [AVAILABLE, DIFFICULT, EXISTING_ONLY, BLOCKED, UNKNOWN]
ASK = "문의"
ALL_DEPARTMENTS = "전체"
ALWAYS = "항상"

LINE_COLUMNS = ["line_no", "target", "departments", "disease_codes", "conditions", "time", "status", "caution",
                "status_source"]
TARGET_COLUMNS = ["hpid", "series_id", "episode_no", "msg_type", "msg_norm", "first_seen", "last_seen", "end_dtm",
                  *LINE_COLUMNS]

_BLOCKED_WORDS = re.compile(r"불가|어렵|어려|제한|중단|없음|없습|부재|부족|휴진")
_CONDITIONAL_WORDS = re.compile(r"가능|문의|상의|확인|연락")


def severe_status(msg: str, is_placeholder: bool = False) -> str:
    """중증 메시지는 떠 있는 것 자체가 불가 신호이므로, 조건을 밝힌 문구만 어려움으로 낮춘다."""
    if is_placeholder or "불가" in msg:
        return BLOCKED
    return DIFFICULT if _CONDITIONAL_WORDS.search(msg) else BLOCKED


def general_status(msg: str) -> str:
    """응급 메시지는 안내문일 수도 있어서 근거 단어가 없으면 판단하지 않는다."""
    if "불가" in msg:
        return BLOCKED
    if _CONDITIONAL_WORDS.search(msg):
        return DIFFICULT
    return BLOCKED if _BLOCKED_WORDS.search(msg) else UNPARSED


def load_parses(path: Path) -> dict[str, dict]:
    """정규화 문구 → LLM 해석 결과. 같은 문구가 여러 번 있으면 마지막 결과를 쓴다."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {record["msg_norm"]: record for record in records}


def usable(answer: dict) -> bool:
    """정답 시트의 답·임시 시험용 답은 그대로 쓴다. 분류 모델 결과는 형식 오류만 없으면 쓴다
    (확신이 낮은 문구는 분류 단계에서 이미 '판단 불가'로 바꿔 둔다). 그 밖(LLM 해석)은 신뢰도가 기준 이상일 때만."""
    if answer.get("source") in ("label", "provisional"):
        return True
    if answer.get("source") == "classifier":
        return not answer.get("errors") and bool(answer.get("lines"))
    return not answer.get("errors") and answer.get("confidence", 0) >= MIN_CONFIDENCE and bool(answer.get("lines"))


def _line(no, target, departments, codes, conditions, time, status, caution, source) -> dict:
    return dict(zip(LINE_COLUMNS, [no, target, departments, codes, conditions, time, status, caution, source]))


def _rule_lines(status: str, departments: str, codes: str, source: str) -> list[dict]:
    caution = ASK if status == DIFFICULT else ""
    return [_line(1, "", departments, codes, "", ALWAYS, status, caution, source)]


def _answer_lines(answer: dict, dept: str) -> list[dict]:
    source = answer.get("source", "llm")
    if not usable(answer):
        return [_line(1, "", dept, "", "", ALWAYS, UNPARSED, "", source)]
    return [_line(i, *line, source) for i, line in enumerate(answer["lines"], 1)]


def build_targets(messages: pd.DataFrame, answers: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    """메시지 1행을 줄 단위 행으로 푼다. answers 는 정규화 문구 → {source, lines, ...} (message_answers.load)."""
    rows = []
    for m in messages.itertuples():
        if m.msg_type == "중증":
            lines = _rule_lines(severe_status(m.msg, m.is_placeholder), ALL_DEPARTMENTS, m.sym_code, "code_rule")
        elif m.msg_norm in answers:
            lines = _answer_lines(answers[m.msg_norm], m.dept)
        else:
            lines = _rule_lines(general_status(m.msg), m.dept, "", "text_rule")
        for line in lines:
            rows.append({
                "hpid": m.hpid, "series_id": m.series_id, "episode_no": m.episode_no, "msg_type": m.msg_type,
                "msg_norm": m.msg_norm, "first_seen": m.first_seen, "last_seen": m.last_seen, "end_dtm": m.end_dtm,
                **line,
            })
    df = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    general = messages.loc[messages["msg_type"] == "응급", "msg_norm"]
    return df, {
        "rows": len(df),
        "by_status": df["status"].value_counts().to_dict(),
        "by_source": df["status_source"].value_counts().to_dict(),
        "general_messages_answered": int(general.isin(answers).sum()),
        "general_messages": int(len(general)),
    }


def build_message_grid(targets: pd.DataFrame, grid_times: pd.Series) -> tuple[pd.DataFrame, dict]:
    """격자 시각 t 에 수집 결과에 떠 있던 메시지 줄을 연결한다.

    first_seen <= t <= last_seen 이고 게시 종료 시각 전이면 떠 있는 것으로 본다.
    종료 시각은 메시지에 함께 게시되므로 t 시점에 알 수 있는 값이다.
    """
    columns = ["hpid", "series_id", "msg_type", *LINE_COLUMNS]
    if targets.empty:
        return pd.DataFrame(columns=["hpid", "t", *columns[1:]]), {"rows": 0, "targets_without_slot": 0}

    slots = pd.Series(grid_times).drop_duplicates().sort_values()
    start = targets["first_seen"].dt.ceil(SLOT).clip(lower=slots.iloc[0])
    latest = targets["last_seen"].where(targets["last_seen"] <= targets["end_dtm"], targets["end_dtm"])
    stop = latest.dt.floor(SLOT).clip(upper=slots.iloc[-1])
    counts = ((stop - start) // SLOT + 1).clip(lower=0).astype(int).to_numpy()

    repeated = targets.loc[np.repeat(targets.index, counts), columns].reset_index(drop=True)
    offsets = np.concatenate([np.arange(n) for n in counts])
    repeated["t"] = start.repeat(counts).reset_index(drop=True) + pd.Series(offsets) * SLOT

    # 같은 시점에 여러 메시지가 뜨면 행은 유지한다. 줄 순서(line_no)는 메시지 안에서만 의미가 있다
    repeated = repeated.sort_values(["hpid", "t", "series_id", "line_no"]).reset_index(drop=True)
    return repeated[["hpid", "t", *columns[1:]]], {
        "rows": len(repeated),
        "targets_without_slot": int((counts == 0).sum()),
    }
