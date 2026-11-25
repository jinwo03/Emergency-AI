"""응급 메시지 문구 정답 세트를 만든다 (문구 목록, 정답 시트, 숨긴 시험용 답).

실행:
    python -m preprocessing.label_set --dump <파일>          # 정답이 필요한 문구 목록만 내보내기
    python -m preprocessing.label_set --answers <jsonl> ...  # 정답 세트 생성 (data/labels/)

규칙
- 대상은 응급 메시지의 고유 문구(정규화 기준)다. 중증 메시지는 질환 코드가 있어 대상이 아니다.
- ID 는 처음 등장한 시각, 같으면 정규화 문구 순으로 매긴다. 이전 세트의 문구는 ID·대표 문구·진료과를 그대로 이어받는다.
  이전 세트는 data/labels/ 에 있는 현재 세트이고, 없으면 --archive 다.
- 시험용은 시간 기준으로 마지막 TEST_RATIO 만큼이다. 같은 시각에 처음 등장한 문구는 같은 쪽에 둔다.
  학습용 문구와 거의 같은 시험용 문구는 옮기지 않고 near_dup 으로 표시만 한다 (문구 목록에만, 시트에는 보이지 않음).
- 현재 시트가 있으면 학습용 정답(사람이 고친 값 포함)과 시험용 입력을 이어받고, 나머지는 새 AI 답으로 채운다.
  덮어쓰기 전에 현재 시트를 .bak.xlsx 로 남긴다.
  --fresh 이면 이어받지 않고 모든 문구를 새 AI 답으로 채운다 ('AI 원래 답'도 새 답, 시험용은 빈칸).
- 시험용 AI 답은 시트에 넣지 않고 claude_test_answers_do_not_open.csv 에만 쓴다.

정답 JSONL 한 줄: {"id": "M0001", "reason": null|사유, "lines": [[대상, 진료과, 질환, 조건, 시간, 상태, 주의], ...], "memo": "..."}
질환·조건·여러 진료과는 ", " 로 잇는다.
"""
import argparse
import csv
import json
import re
from difflib import SequenceMatcher
import shutil
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from contracts.vocab import ALL_DEPARTMENTS, DEPARTMENTS, DISEASE_CODES, PATIENT_CONDITIONS
from egen import store
from preprocessing import clean, label_guide, message_status, time_rule

TEST_RATIO = 150 / 803  # 이전 세트와 같은 비율
NEAR_DUP_SIM = 0.9       # 시험용 문구가 학습용 문구와 이만큼 비슷하면 '중복'으로 표시 (평가 때 따로 봄)
LINES = 4
FIELDS = ["대상", "진료과", "질환", "조건", "시간", "상태", "주의"]
TARGET, DEPT, CODES, CONDS, TIME, STATUS, CAUTION = range(len(FIELDS))
REASONS = ["진료과 없음", "장비·시설", "병상 부족", "의료진 부재", "의료진 부족", "수술·처치 중", "응급실 과밀", "병원 사정"]
STATUSES = message_status.STATUSES
CAUTIONS = ["문의", "지연", "문의·지연"]
ORDER = ["1 시간·여러 대상", "2 기존 환자만", "3 메모 있음", "4 나머지"]
SEP = ", "

PHRASE_FILE = "label_phrases.parquet"
SHEET_FILE = "message_status_labels.xlsx"
HIDDEN_FILE = "claude_test_answers_do_not_open.csv"
PROVISIONAL_FILE = "claude_provisional_test_answers_do_not_open.jsonl"   # 사람 입력 전 임시 시험용 답 (Claude 작성)

TRAIN_SHEET, TEST_SHEET, GUIDE_SHEET, LIST_SHEET = "학습용_확인", "시험용_직접입력", "안내", "목록"
REASON_HEADER = "사유 (환자에게 표시)"
ANSWER_HEADERS = [REASON_HEADER] + [f"{f}{i}" for i in range(1, LINES + 1) for f in FIELDS]
REVIEWED_HEADER, REVIEWED_MARK = "검수", "O"
HOURS_HEADER = "게시 시간"   # 이 문구가 게시돼 있던 시간(병원별 합, 시간). 클수록 추천에 자주 걸린다.
SAMPLE_HEADER, SAMPLE_MARK = "표본", "무작위"   # AI 정답 오류율 추정용 무작위 표본 (맨 위에 모음)
SAMPLE_SIZE, SAMPLE_SEED = 100, 20261122
TRAIN_HEADERS = [REVIEWED_HEADER, SAMPLE_HEADER, "확인 순서", "ID", "문구", *ANSWER_HEADERS, "AI 메모", "메시지 진료과", "등장 수",
                 HOURS_HEADER, "AI 원래 답", *[f"원래 {h}" for h in ANSWER_HEADERS]]
OPTIONAL_TRAIN_HEADERS = (REVIEWED_HEADER, SAMPLE_HEADER, HOURS_HEADER)   # 이전 형식 시트에는 없을 수 있는 칸
LEGACY_TRAIN_HEADERS = [h for h in TRAIN_HEADERS if h not in OPTIONAL_TRAIN_HEADERS]
TEST_HEADERS = ["ID", "메시지 진료과", "문구", *ANSWER_HEADERS, "메모 (선택)"]
FIRST_ANSWER_COL = TRAIN_HEADERS.index(REASON_HEADER) + 1
ANSWER_COLS = range(FIRST_ANSWER_COL, FIRST_ANSWER_COL + len(ANSWER_HEADERS))
TEST_FIRST_ANSWER_COL = TEST_HEADERS.index(REASON_HEADER) + 1
TEST_ANSWER_COLS = range(TEST_FIRST_ANSWER_COL, TEST_FIRST_ANSWER_COL + len(ANSWER_HEADERS))
ORIGINAL_OFFSET = TRAIN_HEADERS.index(f"원래 {REASON_HEADER}") - TRAIN_HEADERS.index(REASON_HEADER)

FIELD_WIDTHS = dict(zip(FIELDS, [16, 11, 9, 11, 18, 11, 6]))
OTHER_WIDTHS = {REVIEWED_HEADER: 5, SAMPLE_HEADER: 7, HOURS_HEADER: 7, "확인 순서": 12, "ID": 8, "문구": 42, REASON_HEADER: 14, "AI 메모": 30, "메시지 진료과": 12,
                "등장 수": 7, "AI 원래 답": 40, "메모 (선택)": 26}
HEADER_FILL = PatternFill("solid", fgColor="FF1F4E78")
INPUT_FILL = PatternFill("solid", fgColor="FFFFFF00")
SECTION_FILL = PatternFill("solid", fgColor="FFDDEBF7")


# ---------- 문구 목록 ----------

def _mode(values: pd.Series) -> str:
    counts = values.value_counts(sort=False)
    top = counts.max()
    return next(v for v in values if counts[v] == top)


def build_phrases(messages: pd.DataFrame, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    general = messages[messages["msg_type"] == "응급"].sort_values(["first_seen", "hpid", "start_dtm"])
    grouped = general.groupby("msg_norm", sort=False)
    df = pd.DataFrame({
        "msg": grouped["msg"].first(),
        "dept": grouped["dept"].agg(_mode),
        "rows": grouped.size(),
        "hospitals": grouped["hpid"].nunique(),
        "first_seen": grouped["first_seen"].min(),
    }).reset_index()

    if previous is not None and not previous.empty:
        prev = previous.set_index("msg_norm")
        known = df["msg_norm"].isin(prev.index)
        for column in ("msg", "dept"):
            df.loc[known, column] = df.loc[known, "msg_norm"].map(prev[column])

    df = df.sort_values(["first_seen", "msg_norm"]).reset_index(drop=True)
    df.insert(0, "id", [f"M{i:04d}" for i in range(1, len(df) + 1)])
    if previous is not None and not previous.empty:
        old_ids = df["msg_norm"].map(previous.set_index("msg_norm")["id"])
        changed = old_ids.notna() & (old_ids != df["id"])
        if changed.any():
            raise ValueError(f"이전 세트의 ID 가 바뀝니다: {df.loc[changed, 'id'].tolist()[:5]}")

    df["rule_status"] = df["msg"].map(message_status.general_status)
    df["split"] = _time_split(df["first_seen"])
    df = df.join(nearest_train(df))
    df["near_dup"] = df["nearest_train_sim"] >= NEAR_DUP_SIM
    return df


# 하나만 달라도 정답이 달라지는 단어. 중복 판정 때 이 단어 집합이 같아야 한다.
KEY_WORDS = sorted({
    *DEPARTMENTS, "내과", "외과", "부인과", "산과", "심장내과", "정신과", "소아과", "응급의학과", "알레르기내과",
    "NS", "OS", "GS", "CS", "PS", "ENT", "OBGY", "ICU", "NICU", "CICU", "중환자실",
    "남자", "여자", "남성", "여성", "소아", "성인", "신생아", "산모", "임산부",
    "두경부", "수부", "족부", "척추", "spine", "vascular", "혈관", "망막", "각막",
    "CPR", "CRP", "f/u", "신환", "포함", "외",
}, key=len, reverse=True)
_KEY_RE = re.compile("|".join(re.escape(w) for w in KEY_WORDS), re.IGNORECASE)


def key_words(text: str) -> frozenset[str]:
    return frozenset(m.group(0).upper().replace("CRP", "CPR") for m in _KEY_RE.finditer(text))


def nearest_train(df: pd.DataFrame) -> pd.DataFrame:
    """시험용 문구마다 핵심 단어(KEY_WORDS) 집합이 같은 학습용 문구 중 가장 비슷한 것의 ID 와 유사도.

    유사도는 정규화 문구의 difflib 비율. 핵심 단어가 같은 학습용 문구가 없거나 학습용 행이면 빈 값.
    """
    train: dict[frozenset, list] = {}
    for tid, ttext in df.loc[df["split"] == "train", ["id", "msg_norm"]].itertuples(index=False):
        train.setdefault(key_words(ttext), []).append((tid, ttext))
    out = pd.DataFrame({"nearest_train_id": pd.Series(None, index=df.index, dtype=object),
                        "nearest_train_sim": pd.Series(float("nan"), index=df.index)})
    for i, text in df.loc[df["split"] == "test", "msg_norm"].items():
        best_id, best = None, 0.0
        matcher = SequenceMatcher(None, b=text, autojunk=False)
        for tid, ttext in train.get(key_words(text), []):
            matcher.set_seq1(ttext)
            if matcher.real_quick_ratio() > best and matcher.quick_ratio() > best:
                ratio = matcher.ratio()
                if ratio > best:
                    best_id, best = tid, ratio
        if best_id:
            out.loc[i] = [best_id, round(best, 3)]
    return out


def _time_split(first_seen: pd.Series) -> pd.Series:
    n_test = round(len(first_seen) * TEST_RATIO)
    boundary = first_seen.iloc[len(first_seen) - n_test]
    return pd.Series(["test" if t >= boundary else "train" for t in first_seen], index=first_seen.index)


# ---------- 정답 ----------

def split_values(cell) -> list[str]:
    return [v.strip() for v in str(cell or "").split(",") if v.strip()]


def validate_answer(answer: dict) -> list[str]:
    """정답 한 건의 형식 오류 목록. 비어 있으면 통과."""
    errors = []
    if answer.get("reason") not in (None, *REASONS):
        errors.append(f"사유 {answer['reason']!r}")
    lines = answer.get("lines") or []
    if not 1 <= len(lines) <= LINES:
        errors.append(f"줄 수 {len(lines)}")
    for i, line in enumerate(lines, 1):
        if len(line) != len(FIELDS):
            errors.append(f"{i}줄 칸 수 {len(line)} (대상, 진료과, 질환, 조건, 시간, 상태, 주의)")
            continue
        if not line[TARGET]:
            errors.append(f"{i}줄 대상 없음")
        depts = split_values(line[DEPT])
        if not depts:
            errors.append(f"{i}줄 진료과 없음")
        elif ALL_DEPARTMENTS in depts and len(depts) > 1:
            errors.append(f"{i}줄 진료과 '{ALL_DEPARTMENTS}'는 혼자 씀")
        errors += [f"{i}줄 진료과 {d!r}" for d in depts if d != ALL_DEPARTMENTS and d not in DEPARTMENTS]
        errors += [f"{i}줄 질환 {c!r}" for c in split_values(line[CODES]) if c not in DISEASE_CODES]
        errors += [f"{i}줄 조건 {c!r}" for c in split_values(line[CONDS]) if c not in PATIENT_CONDITIONS]
        if line[STATUS] not in STATUSES:
            errors.append(f"{i}줄 상태 {line[STATUS]!r}")
        if line[CAUTION] not in (None, "", *CAUTIONS):
            errors.append(f"{i}줄 주의 {line[CAUTION]!r}")
        try:
            time_rule.parse(line[TIME] or "")
        except ValueError as e:
            errors.append(f"{i}줄 시간 {line[TIME]!r}: {e}")
    return errors


def load_answers(paths: list[Path]) -> dict[str, dict]:
    answers = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    answers[record["id"]] = record
    return answers


def answer_cells(answer: dict) -> list:
    """사유 + 4줄 × 7칸 값."""
    cells = [answer.get("reason")]
    lines = answer["lines"]
    for i in range(LINES):
        line = lines[i] if i < len(lines) else [None] * len(FIELDS)
        cells += [v or None for v in line]
    return cells


def answer_summary(answer: dict) -> str:
    return " / ".join(" | ".join(str(v) for v in line if v) for line in answer["lines"])


def review_order(answer: dict) -> str:
    lines = answer["lines"]
    if len(lines) > 1 or any(line[TIME] != time_rule.ALWAYS for line in lines):
        return ORDER[0]
    if any(line[STATUS] == "기존 환자만" for line in lines):
        return ORDER[1]
    return ORDER[2] if answer.get("memo") else ORDER[3]


def load_previous_sheet(sheet_path: Path) -> tuple[dict[str, dict], dict[str, list]]:
    """현재 시트의 학습용 행(ID → {reviewed, sample, order, current, original_text, original, memo})과 시험용 입력(ID → 칸들).

    검수·표본·게시 시간 칸이 없던 이전 형식도 읽는다 (검수 안 됨, 표본 아님으로).
    """
    wb = openpyxl.load_workbook(sheet_path)
    train, test = {}, {}
    ws = wb[TRAIN_SHEET]
    header = [c.value for c in ws[1]]
    while header and header[-1] is None:
        header.pop()
    if [h for h in header if h not in OPTIONAL_TRAIN_HEADERS] != LEGACY_TRAIN_HEADERS:
        raise ValueError(f"{sheet_path} 의 학습용 칸 구성이 지금과 다릅니다. --fresh 로 만드세요.")
    col = {h: i for i, h in enumerate(header)}
    first, n = col[REASON_HEADER], len(ANSWER_HEADERS)
    original = col[f"원래 {REASON_HEADER}"]
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[col["ID"]]:
            mark = r[col[REVIEWED_HEADER]] if REVIEWED_HEADER in col else None
            sample = r[col[SAMPLE_HEADER]] if SAMPLE_HEADER in col else None
            train[r[col["ID"]]] = {
                "reviewed": str(mark or "").strip().upper() == REVIEWED_MARK,
                "sample": sample == SAMPLE_MARK,
                "order": r[col["확인 순서"]],
                "current": list(r[first:first + n]),
                "original_text": r[col["AI 원래 답"]],
                "original": list(r[original:original + n]),
                "memo": r[col["AI 메모"]],
            }
    ws = wb[TEST_SHEET]
    if [c.value for c in ws[1]][:len(TEST_HEADERS)] != TEST_HEADERS:
        raise ValueError(f"{sheet_path} 의 시험용 칸 구성이 지금과 다릅니다. --fresh 로 만드세요.")
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] and any(v not in (None, "") for v in r[3:]):
            test[r[0]] = list(r[3:len(TEST_HEADERS)])
    return train, test


def answer_from_cells(cells: list) -> dict:
    """사유 + 4줄 × 7칸 값(answer_cells 의 반대)을 정답 형식으로. 비어 있는 줄은 뺀다."""
    def text(v) -> str:
        return "" if v is None else str(v).strip()

    n = len(FIELDS)
    lines = [[text(v) for v in cells[1 + i * n:1 + (i + 1) * n]] for i in range(LINES)]
    return {"reason": text(cells[0]) or None, "lines": [line for line in lines if any(line)]}


def load_label_answers(labels_dir: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """정답 시트의 답. (정규화 문구 → {id, split, reviewed, sample, reason, lines}, 형식 오류 ID → 오류 목록).

    학습용은 현재 값(사람이 고친 값 포함, reviewed 는 검수 칸 O), 시험용은 사람이 채운 행만 쓴다(reviewed=True).
    숨긴 시험용 AI 답은 읽지 않는다.
    """
    sheet, phrase_path = labels_dir / SHEET_FILE, labels_dir / PHRASE_FILE
    if not sheet.exists() or not phrase_path.exists():
        return {}, {}
    phrases = pd.read_parquet(phrase_path, columns=["id", "msg_norm", "split"]).set_index("id")
    train, test = load_previous_sheet(sheet)
    cells = {i: row["current"] for i, row in train.items()} | {i: row[:len(ANSWER_HEADERS)] for i, row in test.items()}
    reviewed = {i: row["reviewed"] for i, row in train.items()} | {i: True for i in test}
    sample = {i: row["sample"] for i, row in train.items()}
    answers, invalid = {}, {}
    for pid, values in cells.items():
        if pid not in phrases.index:
            continue
        answer = answer_from_cells(values)
        if errors := validate_answer(answer):
            invalid[pid] = errors
            continue
        p = phrases.loc[pid]
        answers[p["msg_norm"]] = {"id": pid, "split": p["split"], "reviewed": reviewed[pid],
                                  "sample": sample.get(pid, False), **answer}
    return answers, invalid


def load_provisional_answers(labels_dir: Path) -> dict[str, dict]:
    """임시 시험용 답(정규화 문구 → {id, split, reviewed=False, provisional=True, reason, lines}).

    사람이 시험용 시트에 채운 문구는 뺀다. 파일 내용은 화면에 내보내지 않는다 (사람이 보지 않아야 하는 답).
    """
    path, phrase_path = labels_dir / PROVISIONAL_FILE, labels_dir / PHRASE_FILE
    if not path.exists() or not phrase_path.exists():
        return {}
    phrases = pd.read_parquet(phrase_path, columns=["id", "msg_norm", "split"]).set_index("id")
    filled = set(load_previous_sheet(labels_dir / SHEET_FILE)[1]) if (labels_dir / SHEET_FILE).exists() else set()
    out = {}
    for answer in load_answers([path]).values():
        pid = answer["id"]
        if pid in filled or pid not in phrases.index or validate_answer(answer):
            continue
        p = phrases.loc[pid]
        out[p["msg_norm"]] = {"id": pid, "split": p["split"], "reviewed": False, "sample": False,
                              "provisional": True, "reason": answer.get("reason"), "lines": answer["lines"]}
    return out


# ---------- 시트 ----------

def _width(header: str) -> float:
    base = header.removeprefix("원래 ")
    if base in OTHER_WIDTHS:
        return OTHER_WIDTHS[base] if header == base else 8
    return FIELD_WIDTHS[base.rstrip("0123456789")] if header == base else 8


def _write_table(ws, headers: list[str], rows: list[list], input_cols: range | list[int]) -> int:
    ws.append(headers)
    for cell in ws[1]:
        cell.fill, cell.font = HEADER_FILL, Font(bold=True, color="FFFFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 32
    for row in rows:
        ws.append(row)
    last = len(rows) + 1
    wrap = Alignment(wrap_text=True, vertical="top")
    for c, header in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(c)].width = _width(header)
        is_input = c in input_cols
        for r in range(2, last + 1):
            cell = ws.cell(r, c)
            cell.alignment = wrap
            if is_input:
                cell.fill = INPUT_FILL
    ws.freeze_panes = ws.cell(2, headers.index("문구") + 2).coordinate
    return last


def _add_validations(ws, last: int) -> None:
    n_dept, n_code, n_cond = len(DEPARTMENTS) + 1, len(DISEASE_CODES), len(PATIENT_CONDITIONS)
    specs = [
        ([REVIEWED_HEADER], f'"{REVIEWED_MARK}"', "stop"),
        ([REASON_HEADER], '"' + ",".join(REASONS) + '"', "stop"),
        ([f"상태{i}" for i in range(1, LINES + 1)], '"' + ",".join(STATUSES) + '"', "stop"),
        ([f"주의{i}" for i in range(1, LINES + 1)], '"' + ",".join(CAUTIONS) + '"', "stop"),
        ([f"진료과{i}" for i in range(1, LINES + 1)], f"'{LIST_SHEET}'!$A$2:$A${n_dept + 1}", "warning"),
        ([f"질환{i}" for i in range(1, LINES + 1)], f"'{LIST_SHEET}'!$C$2:$C${n_code + 1}", "warning"),
        ([f"조건{i}" for i in range(1, LINES + 1)], f"'{LIST_SHEET}'!$F$2:$F${n_cond + 1}", "warning"),
    ]
    headers = [c.value for c in ws[1]]
    for names, formula, style in specs:
        names = [name for name in names if name in headers]
        if not names:
            continue
        dv = DataValidation(type="list", formula1=formula, allow_blank=True, errorStyle=style,
                            showErrorMessage=True, error="목록에 없는 값입니다. 여러 값은 ', '로 잇습니다.")
        for name in names:
            letter = get_column_letter(headers.index(name) + 1)
            dv.add(f"{letter}2:{letter}{last}")
        ws.add_data_validation(dv)


def _add_change_highlight(ws, last: int) -> None:
    for col in ANSWER_COLS:
        letter, orig = get_column_letter(col), get_column_letter(col + ORIGINAL_OFFSET)
        ws.conditional_formatting.add(
            f"{letter}2:{letter}{last}",
            FormulaRule(formula=[f'${letter}2&""<>${orig}2&""'], font=Font(bold=True, color="FFC00000")),
        )


def _write_lists(ws) -> None:
    ws.append(["진료과", None, "질환 코드", "질환", None, "조건", "뜻"])
    for cell in ws[1]:
        if cell.value:
            cell.fill, cell.font = HEADER_FILL, Font(bold=True, color="FFFFFFFF")
    columns = [
        (1, [ALL_DEPARTMENTS, *DEPARTMENTS]),
        (3, list(DISEASE_CODES)), (4, list(DISEASE_CODES.values())),
        (6, list(PATIENT_CONDITIONS)), (7, list(PATIENT_CONDITIONS.values())),
    ]
    for c, values in columns:
        for r, v in enumerate(values, 2):
            ws.cell(r, c).value = v
    for letter, width in zip("ABCDEFG", [14, 2, 10, 26, 2, 14, 60]):
        ws.column_dimensions[letter].width = width


def _write_guide(ws, n_phrases: int, train_last: int, test_last: int) -> None:
    for letter, width in zip("ABCDEFG", [22, 14, 10, 12, 18, 11, 30]):
        ws.column_dimensions[letter].width = width
    wrap = Alignment(wrap_text=True, vertical="top")
    r = 0
    for kind, *values in label_guide.rows(n_phrases):
        r += 1
        if kind == "title":
            ws.cell(r, 1, values[0]).font = Font(bold=True, size=14)
        elif kind in ("text", "section"):
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
            cell = ws.cell(r, 1, values[0])
            cell.alignment = wrap
            if kind == "section":
                for c in range(1, 8):
                    ws.cell(r, c).fill = SECTION_FILL
                cell.font = Font(bold=True)
            else:
                ws.row_dimensions[r].height = 15 * (1 + len(values[0]) // 110)
        elif kind == "item":
            ws.cell(r, 1, values[0]).font = Font(bold=True)
            ws.cell(r, 1).alignment = wrap
            ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=7)
            ws.cell(r, 2, values[1]).alignment = wrap
            ws.row_dimensions[r].height = 15 * (1 + len(values[1]) // 75)
        elif kind == "row":
            for c, v in enumerate(values[0], 1):
                ws.cell(r, c, v or None)

    tr, te = f"'{TRAIN_SHEET}'", f"'{TEST_SHEET}'"
    changed = "+".join(
        f'({tr}!{get_column_letter(c)}2:{get_column_letter(c)}{train_last}&""<>'
        f'{tr}!{get_column_letter(c + ORIGINAL_OFFSET)}2:{get_column_letter(c + ORIGINAL_OFFSET)}{train_last}&"")'
        for c in ANSWER_COLS)
    status_cols = [get_column_letter(TRAIN_HEADERS.index(f"상태{i}") + 1) for i in range(1, LINES + 1)]
    test_status = get_column_letter(TEST_HEADERS.index("상태1") + 1)
    r += 2
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
    ws.cell(r, 1, "진행 현황").font = Font(bold=True)
    for c in range(1, 8):
        ws.cell(r, c).fill = SECTION_FILL
    progress = [
        ["항목", "입력 수", "전체"],
        ["학습용: 검수 완료 (O)", f'=COUNTIF({tr}!A2:A{train_last},"{REVIEWED_MARK}")', f"=COUNTA({tr}!C2:C{train_last})"],
        ["학습용: 고친 문구", f"=SUMPRODUCT(--(({changed})>0))", f"=COUNTA({tr}!C2:C{train_last})"],
        ["시험용: 입력한 문구", f"=COUNTA({te}!{test_status}2:{test_status}{test_last})", f"=COUNTA({te}!A2:A{test_last})"],
        [],
        ["학습용 정답 분포 (줄 기준)"],
        ["상태", "줄 수"],
    ]
    for values in progress:
        r += 1
        for c, v in enumerate(values, 1):
            ws.cell(r, c, v)
    for status in STATUSES:
        r += 1
        ws.cell(r, 1, status)
        ws.cell(r, 2, "=" + "+".join(f"COUNTIF({tr}!{s}2:{s}{train_last},A{r})" for s in status_cols))


def write_workbook(out: Path, train_rows: list[list], test_rows: list[list], n_phrases: int) -> None:
    wb = openpyxl.Workbook()
    guide = wb.active
    guide.title = GUIDE_SHEET
    tr, te, lists = wb.create_sheet(TRAIN_SHEET), wb.create_sheet(TEST_SHEET), wb.create_sheet(LIST_SHEET)
    _write_lists(lists)

    train_last = _write_table(tr, TRAIN_HEADERS, train_rows, [1, *ANSWER_COLS])
    _add_validations(tr, train_last)
    _add_change_highlight(tr, train_last)
    tr.auto_filter.ref = f"A1:{get_column_letter(TRAIN_HEADERS.index('AI 원래 답') + 1)}{train_last}"
    first_orig = TRAIN_HEADERS.index(f"원래 {REASON_HEADER}") + 1
    tr.column_dimensions.group(get_column_letter(first_orig), get_column_letter(len(TRAIN_HEADERS)), hidden=True)

    test_inputs = list(TEST_ANSWER_COLS) + [len(TEST_HEADERS)]
    test_last = _write_table(te, TEST_HEADERS, test_rows, test_inputs)
    _add_validations(te, test_last)
    te.auto_filter.ref = f"A1:{get_column_letter(len(TEST_HEADERS))}{test_last}"

    _write_guide(guide, n_phrases, train_last, test_last)
    tmp = out.with_name(f"{out.stem}.tmp.xlsx")
    wb.save(tmp)
    tmp.replace(out)


# ---------- 생성 ----------

def _previous_phrases(labels_dir: Path, archive_dir: Path) -> pd.DataFrame:
    path = labels_dir / PHRASE_FILE
    return pd.read_parquet(path if path.exists() else archive_dir / PHRASE_FILE)


def posted_hours(targets: pd.DataFrame) -> dict[str, float]:
    """정규화 문구 → 게시돼 있던 시간 합(병원·게시 단위, 시간). message_targets 의 게시 단위를 쓴다."""
    if targets.empty:
        return {}
    ep = targets.drop_duplicates(["hpid", "series_id", "episode_no", "msg_norm"])
    hours = (ep["last_seen"].fillna(ep["first_seen"]) - ep["first_seen"]).dt.total_seconds() / 3600
    return hours.groupby(ep["msg_norm"]).sum().round(1).to_dict()


def load_posted_hours(processed_dir: Path) -> dict[str, float]:
    path = processed_dir / "message_targets.parquet"
    if not path.exists():
        return {}
    return posted_hours(pd.read_parquet(path, columns=["hpid", "series_id", "episode_no", "msg_norm",
                                                         "first_seen", "last_seen"]))


def draw_sample(train_ids, n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> set[str]:
    """학습용 ID 에서 무작위 n 개 (정렬 후 고정 시드라 다시 뽑아도 같다)."""
    ids = sorted(train_ids)
    rng = np.random.default_rng(seed)
    return set(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist())


def _sample_ids(old_train: dict[str, dict], train_ids) -> set[str]:
    """이전 시트에 표본이 있으면 그대로(검수 기록 유지), 없으면 새로 뽑는다."""
    train_ids = set(train_ids)
    kept = {i for i, o in old_train.items() if o.get("sample") and i in train_ids}
    return kept or draw_sample(train_ids)


def _reused_train_row(old: dict, p, hours: float, sample: bool) -> list:
    return [REVIEWED_MARK if old["reviewed"] else None, SAMPLE_MARK if sample else None, old["order"], p.id, p.msg, *old["current"], old["memo"],
            p.dept, int(p.rows), hours, old["original_text"], *old["original"]]


def _sort_train(rows: list[list]) -> None:
    """무작위 표본을 맨 위에, 그다음 게시 시간이 긴(추천에 자주 걸리는) 문구부터. 같으면 등장 수, ID."""
    sample_col, hours_col, rows_col, id_col = (TRAIN_HEADERS.index(h)
                                               for h in (SAMPLE_HEADER, HOURS_HEADER, "등장 수", "ID"))
    rows.sort(key=lambda r: (r[sample_col] != SAMPLE_MARK, -(r[hours_col] or 0), -r[rows_col], r[id_col]))


def _test_rows(phrases: pd.DataFrame, old_test: dict[str, list], seed: int) -> list[list]:
    test = phrases[phrases["split"] == "test"].sample(frac=1, random_state=seed)
    blank = [None] * (len(TEST_HEADERS) - 3)
    return [[p.id, p.dept, p.msg, *old_test.get(p.id, blank)] for p in test.itertuples()]


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return [round(max(0.0, (centre - half) / denom), 4), round(min(1.0, (centre + half) / denom), 4)]


def _norm_cells(cells: list) -> list[str]:
    return [str(v or "").strip() for v in cells]


def audit(labels_dir: Path) -> dict:
    """검수 진행 상황과, 검수한 무작위 표본에서 AI 정답이 고쳐진 비율(=AI 정답 오류율 추정, Wilson 95% 구간)."""
    train, test = load_previous_sheet(labels_dir / SHEET_FILE)
    status_idx = [1 + i * len(FIELDS) + STATUS for i in range(LINES)]

    def changed(o: dict) -> bool:
        return _norm_cells(o["current"]) != _norm_cells(o["original"])

    def status_changed(o: dict) -> bool:
        cur, org = _norm_cells(o["current"]), _norm_cells(o["original"])
        return [cur[i] for i in status_idx] != [org[i] for i in status_idx]

    done = [o for o in train.values() if o["sample"] and o["reviewed"]]
    k, ks, n = sum(map(changed, done)), sum(map(status_changed, done)), len(done)
    n_test = int((pd.read_parquet(labels_dir / PHRASE_FILE, columns=["split"])["split"] == "test").sum())
    return {
        "train_reviewed": sum(o["reviewed"] for o in train.values()), "train_total": len(train),
        "train_changed": sum(map(changed, train.values())),
        "test_filled": len(test), "test_total": n_test,
        "sample_reviewed": n, "sample_total": sum(o["sample"] for o in train.values()),
        "ai_label_error_rate": {
            "any_cell": round(k / n, 4) if n else None, "any_cell_ci95": _wilson(k, n),
            "status": round(ks / n, 4) if n else None, "status_ci95": _wilson(ks, n),
        },
    }


def refresh(labels_dir: Path, processed_dir: Path, seed: int = 20261119) -> dict:
    """DB 없이 현재 시트를 지금 형식으로 다시 쓴다. 문구·분할·답·검수 표시는 그대로, 게시 시간만 새로 계산.

    숨긴 시험용 답 파일은 건드리지 않는다.
    """
    sheet = labels_dir / SHEET_FILE
    phrases = pd.read_parquet(labels_dir / PHRASE_FILE)
    old_train, old_test = load_previous_sheet(sheet)
    hours = load_posted_hours(processed_dir)
    train = phrases[phrases["split"] == "train"]
    missing = sorted(set(train["id"]) - set(old_train))
    if missing:
        raise ValueError(f"현재 시트에 없는 학습용 문구 {len(missing)}건 {missing[:10]}")
    sample = _sample_ids(old_train, train["id"])
    train_rows = [_reused_train_row(old_train[p.id], p, hours.get(p.msg_norm, 0.0), p.id in sample)
                  for p in train.itertuples()]
    _sort_train(train_rows)
    test_rows = _test_rows(phrases, old_test, seed)
    shutil.copyfile(sheet, sheet.with_name(f"{sheet.stem}.bak.xlsx"))
    write_workbook(sheet, train_rows, test_rows, len(phrases))
    return {"train": len(train_rows), "test": len(test_rows),
            "reviewed": sum(o["reviewed"] for o in old_train.values()), "sample": len(sample),
            "test_inputs_kept": len(old_test),
            "with_hours": sum(1 for p in train.itertuples() if p.msg_norm in hours)}


def build(labels_dir: Path, archive_dir: Path, answer_paths: list[Path], seed: int = 20261119,
          reuse_previous: bool = True, processed_dir: Path | None = None) -> dict:
    previous = _previous_phrases(labels_dir, archive_dir)
    with clean.open_readonly() as conn:
        messages, _ = clean.clean_messages(conn)
    phrases = build_phrases(messages, previous)
    sheet = labels_dir / SHEET_FILE
    old_train, old_test = load_previous_sheet(sheet) if reuse_previous and sheet.exists() else ({}, {})
    answers = load_answers(answer_paths)
    hours = load_posted_hours(processed_dir) if processed_dir else {}

    train_rows, hidden, missing, invalid = [], [], [], {}
    sample = _sample_ids(old_train, phrases.loc[phrases["split"] == "train", "id"])
    for p in phrases.itertuples():
        reused = p.split == "train" and p.id in old_train
        if not reused:
            answer = answers.get(p.id)
            if answer is None:
                missing.append(p.id)
                continue
            if errors := validate_answer(answer):
                invalid[p.id] = errors
                continue
        if p.split == "test":
            hidden.append([p.id, p.msg, *[v or "" for v in answer_cells(answer)], answer.get("memo") or ""])
        elif reused:
            train_rows.append(_reused_train_row(old_train[p.id], p, hours.get(p.msg_norm, 0.0), p.id in sample))
        else:
            cells = answer_cells(answer)
            train_rows.append([None, SAMPLE_MARK if p.id in sample else None, review_order(answer), p.id, p.msg, *cells, answer.get("memo"), p.dept,
                               int(p.rows), hours.get(p.msg_norm, 0.0), answer_summary(answer), *cells])
    if missing or invalid:
        raise ValueError(f"정답 없음 {len(missing)}건 {missing[:10]}, 형식 오류 {len(invalid)}건 {dict(list(invalid.items())[:10])}")

    _sort_train(train_rows)
    test = phrases[phrases["split"] == "test"]
    test_rows = _test_rows(phrases, old_test, seed)

    labels_dir.mkdir(parents=True, exist_ok=True)
    if sheet.exists():
        shutil.copyfile(sheet, sheet.with_name(f"{sheet.stem}.bak.xlsx"))
    phrases.to_parquet(labels_dir / PHRASE_FILE, index=False)
    write_workbook(sheet, train_rows, test_rows, len(phrases))
    with (labels_dir / HIDDEN_FILE).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "문구", *ANSWER_HEADERS, "메모"])
        writer.writerows(sorted(hidden))

    reused = sum(1 for p in phrases.itertuples() if p.split == "train" and p.id in old_train)
    return {
        "phrases": len(phrases),
        "train": len(train_rows),
        "test": len(test_rows),
        "train_reused_from_previous": reused,
        "train_new_ai_answers": len(train_rows) - reused,
        "test_inputs_kept": sum(1 for p in test.itertuples() if p.id in old_test),
        "test_hidden_answers": len(hidden),
        "test_near_dup": int(phrases["near_dup"].sum()),
        "test_first_seen_from": str(phrases.loc[phrases["split"] == "test", "first_seen"].min()),
    }


def dump(path: Path, labels_dir: Path, archive_dir: Path, everything: bool = False) -> int:
    """정답이 필요한 문구(현재 학습용에 없는 것, everything 이면 전부)를 JSONL 로 내보낸다."""
    previous = _previous_phrases(labels_dir, archive_dir)
    with clean.open_readonly() as conn:
        messages, _ = clean.clean_messages(conn)
    phrases = build_phrases(messages, previous)
    sheet = labels_dir / SHEET_FILE
    reusable = set() if everything or not sheet.exists() else set(load_previous_sheet(sheet)[0])
    todo = phrases[~((phrases["split"] == "train") & phrases["id"].isin(reusable))]
    with path.open("w", encoding="utf-8") as f:
        for p in todo.itertuples():
            f.write(json.dumps({"id": p.id, "split": p.split, "dept": p.dept, "rows": int(p.rows), "msg": p.msg},
                               ensure_ascii=False) + "\n")
    return len(todo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=store.DEFAULT_DB.parent / "labels")
    parser.add_argument("--archive", type=Path, default=store.DEFAULT_DB.parent / "labels" / "archive_2026-11-06",
                        help="현재 세트가 없을 때 ID 를 이어받을 이전 세트")
    parser.add_argument("--dump", type=Path, help="정답이 필요한 문구 목록을 이 파일로 내보내기")
    parser.add_argument("--answers", type=Path, nargs="*", default=[], help="AI 정답 JSONL 파일들")
    parser.add_argument("--fresh", action="store_true",
                        help="현재 시트를 이어받지 않고 모든 문구를 --answers 로 채운다")
    parser.add_argument("--processed", type=Path, default=store.DEFAULT_DB.parent / "processed",
                        help="게시 시간 계산에 쓸 message_targets.parquet 위치")
    parser.add_argument("--audit", action="store_true",
                        help="검수 진행 상황과 무작위 표본 기준 AI 정답 오류율")
    parser.add_argument("--refresh", action="store_true",
                        help="DB 없이 현재 시트를 지금 형식으로 다시 쓰기 (답·검수 표시 유지, 게시 시간 갱신)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.audit:
        print(json.dumps(audit(args.labels), ensure_ascii=False, indent=2))
        return
    if args.refresh:
        print(json.dumps(refresh(args.labels, args.processed), ensure_ascii=False, indent=2))
        return
    if args.dump:
        print(f"{dump(args.dump, args.labels, args.archive, everything=args.fresh)}건 → {args.dump}")
        return
    print(json.dumps(build(args.labels, args.archive, args.answers, reuse_previous=not args.fresh,
                           processed_dir=args.processed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
