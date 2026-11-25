"""LLM 해석 결과를 정답 시트와 비교해 점수를 낸다.

실행:
    python -m preprocessing.score_parses                                  # data/labels/message_parse.jsonl 채점
    python -m preprocessing.score_parses --pred <jsonl> --prompt-version <버전> --details <csv>

정답: 학습용은 시트의 현재 값, 시험용은 사람이 채운 행만 (숨긴 시험용 AI 답은 쓰지 않음).
학습용 점수는 지시문을 고치는 데, 시험용 점수는 최종 평가에 쓴다. 시험용은 near_dup(학습용과 거의 같은 문구)을
뺀 점수도 따로 낸다.

지표 (문구마다 계산해 평균)
- exact: 줄 수와 모든 줄의 진료과·질환·조건·시간·상태·주의가 순서까지 같음 (대상 칸은 사람용이라 비교하지 않음)
- reason: 사유가 같음 / line_count: 줄 수가 같음
- fields: 줄을 짝지은 뒤(같은 칸이 많은 짝부터) 칸별 일치율. 짝이 없는 줄은 틀린 것으로 센다
- patient_status: 줄에 나온 진료과·질환·조건으로 만든 가상 환자 × 한 주(30분 간격) 시각마다
  정답과 해석의 최종 상태가 같은 비율 (algorithm.message_match)
- patient_blocked: 같은 방식으로 '못 받음(불가·기존 환자만)' 여부만 비교
- patient_caution: 최종 주의(문의·지연)가 같은 비율
형식 오류가 있거나 해석이 없는 문구는 점수에서 빼고 invalid / missing 으로 센다.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from algorithm.message_match import CAUTION, CODES, CONDS, DEPT, STATUS, TIME, Patient, status_for, values
from contracts.vocab import ALL_DEPARTMENTS
from egen import store
from preprocessing import label_set, message_status

COMPARED = [DEPT, CODES, CONDS, TIME, STATUS, CAUTION]
FIELD_NAMES = {i: label_set.FIELDS[i] for i in COMPARED}
BLOCKED = {message_status.BLOCKED, message_status.EXISTING_ONLY}
OTHER_DEPT = "(목록에 없는 진료과)"
WEEK_START = datetime(2026, 11, 16)  # 공휴일 없는 월요일
TIMES = [WEEK_START + timedelta(minutes=30 * i) for i in range(7 * 48)]
METRICS = ["exact", "reason", "line_count", "patient_status", "patient_blocked", "patient_caution"]


def _cell(line: list, i: int):
    if i in (DEPT, CODES, CONDS):
        return values(line[i])
    return (line[i] or "").strip()


def norm_line(line: list) -> tuple:
    return tuple(_cell(line, i) for i in COMPARED)


def align(gold: list[list], pred: list[list]) -> list[tuple[int | None, int | None]]:
    """같은 칸이 많은 짝부터 줄을 짝짓는다. 짝이 없으면 한쪽이 None."""
    scores = sorted(((sum(a == b for a, b in zip(norm_line(g), norm_line(p))), -abs(i - j), i, j)
                     for i, g in enumerate(gold) for j, p in enumerate(pred)), reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for _, _, i, j in scores:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            pairs.append((i, j))
    pairs += [(i, None) for i in range(len(gold)) if i not in used_g]
    pairs += [(None, j) for j in range(len(pred)) if j not in used_p]
    return pairs


def patients(*answers: list[list]) -> list[Patient]:
    """줄마다 그 줄에 딱 맞는 환자, 질환·조건을 뺀 환자, 그리고 목록에 없는 진료과 환자."""
    out = {Patient.of([OTHER_DEPT])}
    for lines in answers:
        for line in lines:
            depts = values(line[DEPT]) or {ALL_DEPARTMENTS}
            codes, conds = values(line[CODES]), values(line[CONDS])
            for d in depts:
                dept = OTHER_DEPT if d == ALL_DEPARTMENTS else d
                for c in [frozenset(), *({x} for x in codes)]:
                    for k in {frozenset(), conds}:
                        out.add(Patient.of([dept], c, k))
    return sorted(out, key=lambda p: (sorted(p.departments), sorted(p.disease_codes), sorted(p.conditions)))


def score_phrase(gold: dict, pred: dict) -> dict:
    g, p = gold["lines"], pred["lines"]
    pairs = align(g, p)
    fields = {}
    for i in COMPARED:
        hits = [gi is not None and pj is not None and _cell(g[gi], i) == _cell(p[pj], i) for gi, pj in pairs]
        fields[FIELD_NAMES[i]] = sum(hits) / len(hits)

    same_status = same_blocked = same_caution = total = 0
    for patient in patients(g, p):
        for at in TIMES:
            gs, gc, _ = status_for(g, patient, at)
            ps, pc, _ = status_for(p, patient, at)
            same_status += gs == ps
            same_blocked += (gs in BLOCKED) == (ps in BLOCKED)
            same_caution += gc == pc
            total += 1
    return {
        "exact": len(g) == len(p) and all(norm_line(a) == norm_line(b) for a, b in zip(g, p)),
        "reason": (gold.get("reason") or None) == (pred.get("reason") or None),
        "line_count": len(g) == len(p),
        **{f"field_{k}": v for k, v in fields.items()},
        "patient_status": same_status / total,
        "patient_blocked": same_blocked / total,
        "patient_caution": same_caution / total,
    }


def load_predictions(path: Path, prompt_version: str | None = None) -> dict[str, dict]:
    """정규화 문구 → 해석. 같은 문구가 여러 번 있으면 (지정한 지시문 버전 중) 마지막 것."""
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if prompt_version is None or record.get("prompt_version") == prompt_version:
                    out[record["msg_norm"]] = record
    return out


def load_gold(labels_dir: Path, provisional: bool = False) -> pd.DataFrame:
    """정답 시트의 답 + near_dup. 열: id, msg_norm, split, reviewed, sample, provisional, near_dup, answer.

    provisional 이면 사람이 아직 채우지 않은 시험용 문구에 임시 답(Claude 작성)을 쓴다.
    """
    answers, _ = label_set.load_label_answers(labels_dir)
    if provisional:
        answers = label_set.load_provisional_answers(labels_dir) | answers
    near = pd.read_parquet(labels_dir / label_set.PHRASE_FILE, columns=["id", "near_dup"]).set_index("id")["near_dup"]
    rows = [{"id": a["id"], "msg_norm": k, "split": a["split"], "reviewed": bool(a.get("reviewed", False)),
             "sample": bool(a.get("sample", False)), "provisional": bool(a.get("provisional", False)),
             "near_dup": bool(near.get(a["id"], False)), "answer": a} for k, a in answers.items()]
    return pd.DataFrame(rows, columns=["id", "msg_norm", "split", "reviewed", "sample", "provisional", "near_dup", "answer"])


def _summary(df: pd.DataFrame, gold_n: int) -> dict:
    scored = df[df["status"] == "scored"]
    out = {"gold": gold_n, "scored": len(scored), "missing": int((df["status"] == "missing").sum()),
           "invalid": int((df["status"] == "invalid").sum())}
    if not scored.empty:
        columns = METRICS + [c for c in scored.columns if c.startswith("field_")]
        out |= {c: round(float(scored[c].mean()), 4) for c in columns}
    return out


def score(gold: pd.DataFrame, preds: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    rows = []
    for g in gold.itertuples():
        pred = preds.get(g.msg_norm)
        row = {"id": g.id, "split": g.split, "reviewed": getattr(g, "reviewed", False),
               "sample": getattr(g, "sample", False), "provisional": getattr(g, "provisional", False), "near_dup": g.near_dup}
        if pred is None:
            row["status"] = "missing"
        elif pred.get("errors") or label_set.validate_answer(pred):
            row["status"] = "invalid"
        else:
            row |= {"status": "scored", **score_phrase(g.answer, pred)}
        rows.append(row)
    details = pd.DataFrame(rows)
    report = {}
    for split in ("train", "test"):
        part = details[details["split"] == split]
        if part.empty:
            continue
        report[split] = _summary(part, len(part))
        if split == "train":
            checked = part[part["reviewed"].astype(bool)]
            report["train_reviewed"] = _summary(checked, len(checked))   # 사람이 검수(O)한 답만
            random_checked = checked[checked["sample"].astype(bool)]
            report["train_sample_reviewed"] = _summary(random_checked, len(random_checked))   # 무작위 표본 중 검수분
        if split == "test":
            report["test_gold_provisional"] = int(part["provisional"].astype(bool).sum())   # 임시 답으로 채점한 수
            fresh = part[~part["near_dup"]]
            report["test_without_near_dup"] = _summary(fresh, len(fresh))
    return report, details


def main() -> None:
    labels = store.DEFAULT_DB.parent / "labels"
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=labels)
    parser.add_argument("--pred", type=Path, default=labels / message_status.PARSE_FILE)
    parser.add_argument("--prompt-version", help="이 지시문 버전의 해석만 채점")
    parser.add_argument("--details", type=Path, help="문구별 점수를 이 CSV 로 저장")
    parser.add_argument("--provisional", action="store_true",
                        help="사람이 채우지 않은 시험용 문구는 임시 답(Claude 작성)으로 채점")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    report, details = score(load_gold(args.labels, args.provisional), load_predictions(args.pred, args.prompt_version))
    if args.details:
        details.sort_values(["split", "patient_status"] if "patient_status" in details else ["split"]).to_csv(
            args.details, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
