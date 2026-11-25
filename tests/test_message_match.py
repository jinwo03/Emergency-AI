from datetime import datetime

import pandas as pd
import pytest

from algorithm.message_match import Patient, line_applies, status_for
from preprocessing import score_parses

MON_10 = datetime(2026, 11, 16, 10, 0)
MON_23 = datetime(2026, 11, 16, 23, 0)
SAT_10 = datetime(2026, 11, 21, 10, 0)


def line(target="x", dept="외과", codes="", conds="", time="항상", status="불가", caution="") -> list:
    return [target, dept, codes, conds, time, status, caution]


@pytest.mark.parametrize("ln, patient, expected", [
    (line(dept="외과"), Patient.of(["외과"]), True),
    (line(dept="외과, 정형외과"), Patient.of(["정형외과"]), True),
    (line(dept="외과"), Patient.of(["신경과"]), False),
    (line(dept="전체"), Patient.of(["신경과"]), True),
    (line(dept="전체", codes="Y0020"), Patient.of(["신경과"]), False),
    (line(dept="전체", codes="Y0020, Y0010"), Patient.of(["신경과"], ["Y0010"]), True),
    (line(dept="전체", conds="소아, 중증"), Patient.of(["외과"], conditions=["소아"]), False),
    (line(dept="전체", conds="소아, 중증"), Patient.of(["외과"], conditions=["소아", "중증", "외상"]), True),
])
def test_line_applies(ln, patient, expected):
    assert line_applies(ln, patient) is expected


def test_first_matching_line_wins_and_default_is_available():
    lines = [line("뇌경색", "신경과", "Y0020", status="가능"), line("신경과", "신경과", time="23:00-08:00")]
    stroke, dizzy = Patient.of(["신경과"], ["Y0020"]), Patient.of(["신경과"])
    assert status_for(lines, stroke, MON_23)[:2] == ("가능", "")
    assert status_for(lines, dizzy, MON_23) == ("불가", "", 1)
    assert status_for(lines, dizzy, MON_10) == ("가능", "", None)
    assert status_for(lines, Patient.of(["외과"]), MON_23) == ("가능", "", None)


def test_otherwise_uses_lines_with_same_key_only():
    lines = [line(dept="정형외과", time="평일 08:00-21:00", status="가능", caution="문의"),
             line(dept="정형외과", time="그 외", status="기존 환자만"),
             line(dept="전체", status="어려움")]
    p = Patient.of(["정형외과"])
    assert status_for(lines, p, MON_10)[:2] == ("가능", "문의")
    assert status_for(lines, p, SAT_10)[:2] == ("기존 환자만", "")
    assert status_for(lines, Patient.of(["외과"]), SAT_10)[:2] == ("어려움", "")


def test_patients_cover_each_line():
    ps = score_parses.patients([line(dept="전체", codes="Y0070", conds="소아"), line(dept="외과, 흉부외과")])
    depts = {d for p in ps for d in p.departments}
    assert depts == {score_parses.OTHER_DEPT, "외과", "흉부외과"}
    assert Patient.of([score_parses.OTHER_DEPT], ["Y0070"], ["소아"]) in ps


def test_align_pairs_most_similar_lines():
    gold = [line(dept="외과"), line(dept="안과", status="가능")]
    pred = [line(dept="안과", status="가능"), line(dept="외과"), line(dept="치과")]
    assert set(score_parses.align(gold, pred)) == {(0, 1), (1, 0), (None, 2)}


def answer(*lines, reason=None) -> dict:
    return {"reason": reason, "lines": list(lines)}


def test_score_phrase_identical_is_perfect():
    a = answer(line("뇌경색", "신경과", "Y0020", status="가능"), line(dept="신경과", time="야간"))
    result = score_parses.score_phrase(a, a)
    assert all(v == 1 for v in result.values())


def test_score_phrase_wrong_time_lowers_patient_score_only_partly():
    gold = answer(line(dept="신경과", time="23:00-08:00"))
    pred = answer(line(dept="신경과", time="야간"))
    result = score_parses.score_phrase(gold, pred)
    assert not result["exact"] and result["field_시간"] == 0 and result["field_상태"] == 1
    assert 0.5 < result["patient_status"] < 1


def test_score_splits_and_counts_missing_invalid():
    gold = pd.DataFrame([
        {"id": "M1", "msg_norm": "a", "split": "train", "near_dup": False, "answer": answer(line())},
        {"id": "M2", "msg_norm": "b", "split": "test", "near_dup": True, "answer": answer(line())},
        {"id": "M3", "msg_norm": "c", "split": "test", "near_dup": False, "answer": answer(line())},
        {"id": "M4", "msg_norm": "d", "split": "test", "near_dup": False, "answer": answer(line())},
    ])
    preds = {"a": answer(line()), "b": answer(line(status="가능")), "c": {**answer(line(time="밤")), "errors": ["x"]}}
    report, details = score_parses.score(gold, preds)
    assert report["train"]["exact"] == 1.0
    assert report["test"]["scored"] == 1 and report["test"]["exact"] == 0.0
    assert (report["test"]["missing"], report["test"]["invalid"]) == (1, 1)
    assert report["test_without_near_dup"]["scored"] == 0
    assert set(details["status"]) == {"scored", "missing", "invalid"}


def test_triage_departments_all_reach_message_vocabulary():
    """증상 분석이 낼 수 있는 진료과는 모두 메시지 진료과로 대조할 수 있어야 한다 (안전 필터가 빠지지 않게)."""
    from algorithm import egen_codes
    from contracts import vocab

    known = set(vocab.DEPARTMENTS) | set(vocab.TRIAGE_ONLY_DEPARTMENTS)
    assert set(egen_codes.DEPARTMENTS) <= known
    assert {d for c in egen_codes.EGEN_CONDITIONS.values() for d in c["depts"]} <= known
    assert set(egen_codes.EGEN_CONDITIONS) == set(vocab.DISEASE_CODES)


def test_general_internal_medicine_matches_subspecialty_message():
    from algorithm.message_match import Patient, line_applies

    line = ["소화기 내시경", "소화기내과", "", "", "항상", "불가", ""]
    assert line_applies(line, Patient.of(["내과"]))
    assert line_applies(["대동맥", "흉부외과", "", "", "항상", "불가", ""], Patient.of(["심장혈관흉부외과"]))
    assert not line_applies(line, Patient.of(["응급의학과"]))
