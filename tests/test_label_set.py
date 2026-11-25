import openpyxl
import pandas as pd
import pytest

from preprocessing import label_set


def ts(*minutes) -> pd.Series:
    return pd.Series(pd.to_datetime([f"2026-11-01T10:{m:02d}:00+09:00" for m in minutes]))


def messages(rows) -> pd.DataFrame:
    """rows: (msg_norm, msg, dept, hpid, first_seen 분)"""
    df = pd.DataFrame(rows, columns=["msg_norm", "msg", "dept", "hpid", "minute"])
    df["first_seen"] = ts(*df.pop("minute"))
    df["start_dtm"] = df["first_seen"]
    df["msg_type"] = "응급"
    return df


def line(target="전체", dept="전체", codes="", conds="", spec="항상", status="불가", caution=""):
    return [target, dept, codes, conds, spec, status, caution]


def test_validate_answer_accepts_sheet_values():
    answer = {"reason": "의료진 부재", "lines": [
        line("외과", "외과", spec="평일 주간", status="가능", caution="문의"),
        line("소아 장중첩", "전체", "Y0070", "소아, 중증", "그 외"),
        line("성형외과·안과", "성형외과, 안과"),
    ]}
    assert label_set.validate_answer(answer) == []


@pytest.mark.parametrize("answer, problem", [
    ({"reason": None, "lines": [line(status="조건부")]}, "상태"),
    ({"reason": None, "lines": [line(spec="평일 8시")]}, "시간"),
    ({"reason": "기타", "lines": [line()]}, "사유"),
    ({"reason": None, "lines": [line(caution="확인")]}, "주의"),
    ({"reason": None, "lines": []}, "줄 수"),
    ({"reason": None, "lines": [line()] * 5}, "줄 수"),
    ({"reason": None, "lines": [line(dept="")]}, "진료과 없음"),
    ({"reason": None, "lines": [line(dept="심장내과")]}, "진료과"),
    ({"reason": None, "lines": [line(dept="전체, 외과")]}, "혼자"),
    ({"reason": None, "lines": [line(codes="AMI")]}, "질환"),
    ({"reason": None, "lines": [line(conds="노인")]}, "조건"),
    ({"reason": None, "lines": [["전체", "항상", "불가", ""]]}, "칸 수"),
])
def test_validate_answer_rejects(answer, problem):
    assert any(problem in e for e in label_set.validate_answer(answer))


def test_workbook_round_trip(tmp_path):
    answer = {"reason": None, "lines": [line("CPR 환자", conds="CPR", spec="야간"), line("외과", "외과", status="가능")],
              "memo": "m"}
    cells = label_set.answer_cells(answer)
    train = [["O", "무작위", label_set.review_order(answer), "M0001", "문구", *cells, "m", "응급실", 3, 12.5,
              label_set.answer_summary(answer), *cells],
             [None, None, label_set.review_order(answer), "M0003", "문구3", *cells, None, "외과", 1, 0.0,
              label_set.answer_summary(answer), *cells]]
    test = [["M0002", "외과", "문구2", *[None] * (len(label_set.TEST_HEADERS) - 3)]]
    out = tmp_path / "sheet.xlsx"
    label_set.write_workbook(out, train, test, 3)
    old_train, old_test = label_set.load_previous_sheet(out)
    assert old_train["M0001"]["current"] == old_train["M0001"]["original"] == cells
    assert old_train["M0001"]["memo"] == "m"
    assert old_train["M0001"]["reviewed"] and not old_train["M0003"]["reviewed"]
    assert old_train["M0001"]["sample"] and not old_train["M0003"]["sample"]
    assert old_test == {}


def test_reads_legacy_sheet_without_review_columns(tmp_path):
    answer = {"reason": None, "lines": [line("외과", "외과", status="가능")], "memo": None}
    cells = label_set.answer_cells(answer)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = label_set.TRAIN_SHEET
    ws.append(label_set.LEGACY_TRAIN_HEADERS)
    ws.append([label_set.ORDER[3], "M0001", "문구", *cells, None, "외과", 2, "요약", *cells])
    te = wb.create_sheet(label_set.TEST_SHEET)
    te.append(label_set.TEST_HEADERS)
    out = tmp_path / "old.xlsx"
    wb.save(out)
    old_train, _ = label_set.load_previous_sheet(out)
    assert old_train["M0001"]["current"] == old_train["M0001"]["original"] == cells
    assert old_train["M0001"]["reviewed"] is False and old_train["M0001"]["sample"] is False


def test_draw_sample_is_stable_and_kept():
    ids = [f"M{i:04d}" for i in range(300)]
    a = label_set.draw_sample(ids, n=20)
    assert len(a) == 20 and a == label_set.draw_sample(list(reversed(ids)), n=20)
    old = {i: {"sample": i in {"M0001", "M0002"}} for i in ids}
    assert label_set._sample_ids(old, ids) == {"M0001", "M0002"}


def test_wilson_interval():
    lo, hi = label_set._wilson(10, 100)
    assert 0.05 < lo < 0.1 < hi < 0.18


def test_posted_hours_sums_episodes_per_phrase():
    t0 = pd.Timestamp("2026-11-01 00:00", tz="Asia/Seoul")
    targets = pd.DataFrame({
        "hpid": ["H1", "H1", "H2", "H2"], "series_id": [1, 1, 2, 2], "episode_no": [1, 1, 1, 2],
        "msg_norm": ["a", "a", "a", "b"],                       # H1 의 a 는 줄이 2개여도 한 번만
        "first_seen": [t0] * 4,
        "last_seen": [t0 + pd.Timedelta(hours=3)] * 2 + [t0 + pd.Timedelta(hours=1), pd.NaT],
    })
    assert label_set.posted_hours(targets) == {"a": 4.0, "b": 0.0}


def test_time_split_keeps_same_first_seen_together(monkeypatch):
    monkeypatch.setattr(label_set, "TEST_RATIO", 0.25)
    split = label_set._time_split(ts(0, 1, 2, 2, 3, 4, 5, 6))   # 뒤 2개가 시험용, 경계는 같은 시각 없음
    assert split.tolist() == ["train"] * 6 + ["test"] * 2
    split = label_set._time_split(ts(0, 1, 2, 3, 4, 5, 6, 6))   # 경계 시각이 같으면 함께 시험용
    assert split.tolist() == ["train"] * 6 + ["test"] * 2
    split = label_set._time_split(ts(0, 1, 2, 3, 4, 5, 5, 6))
    assert split.tolist() == ["train"] * 5 + ["test"] * 3


def test_build_phrases_keeps_previous_ids_and_text():
    msgs = messages([
        ("a", "A 원문", "외과", "H1", 0),
        ("a", "A 원문2", "내과", "H2", 5),
        ("b", "B", "안과", "H1", 1),
        ("c", "C", "치과", "H3", 9),
    ])
    previous = pd.DataFrame({"id": ["M0001", "M0002"], "msg_norm": ["a", "b"], "msg": ["A 이전", "B"],
                             "dept": ["이전과", "안과"]})
    df = label_set.build_phrases(msgs, previous).set_index("msg_norm")
    assert df.loc["a", ["id", "msg", "dept", "rows", "hospitals"]].tolist() == ["M0001", "A 이전", "이전과", 2, 2]
    assert df.loc["c", "id"] == "M0003"


def test_build_phrases_refuses_to_renumber_previous():
    msgs = messages([("a", "A", "외과", "H1", 5), ("b", "B", "외과", "H1", 1)])
    previous = pd.DataFrame({"id": ["M0001"], "msg_norm": ["a"], "msg": ["A"], "dept": ["외과"]})
    with pytest.raises(ValueError):
        label_set.build_phrases(msgs, previous)


def test_review_order():
    assert label_set.review_order({"lines": [line(spec="야간")]}) == "1 시간·여러 대상"
    assert label_set.review_order({"lines": [line(status="기존 환자만")]}) == "2 기존 환자만"
    assert label_set.review_order({"lines": [line()], "memo": "x"}) == "3 메모 있음"
    assert label_set.review_order({"lines": [line()]}) == "4 나머지"


def test_near_dup_marks_only_test_rows_similar_to_train():
    df = pd.DataFrame({"id": ["M1", "M2", "M3", "M4"],
                       "msg_norm": ["야간 안과 진료 불가", "소아 진료 가능", "야간 안과 진료 불가.", "정형외과 응급수술 불가"],
                       "split": ["train", "train", "test", "test"]})
    near = label_set.nearest_train(df)
    assert pd.isna(near.loc[0, "nearest_train_id"]) and pd.isna(near.loc[0, "nearest_train_sim"])
    assert near.loc[2, "nearest_train_id"] == "M1" and near.loc[2, "nearest_train_sim"] >= label_set.NEAR_DUP_SIM
    assert not near.loc[3, "nearest_train_sim"] >= label_set.NEAR_DUP_SIM


def test_near_dup_requires_same_key_words():
    df = pd.DataFrame({"id": ["M1", "M2", "M3", "M4"],
                       "msg_norm": ["[신장내과] 진료 불가합니다. f/u 환자는 문의 주세요.", "현재 본원 여자 병실이 없습니다.",
                                    "[심장내과] 진료 불가합니다. f/u 환자는 문의 주세요.", "현재 본원 남자 병실이 없습니다"],
                       "split": ["train", "train", "test", "test"]})
    near = label_set.nearest_train(df)
    assert near.loc[2:, "nearest_train_id"].isna().all()
    assert label_set.key_words("응급실포화, CRP제외") == label_set.key_words("응급실 포화 CPR제외")


def test_provisional_answers_yield_to_human_test_input(tmp_path):
    import json
    from preprocessing import message_answers
    ok = {"reason": None, "lines": [line("외과", "외과", status="가능")], "memo": None}
    pd.DataFrame({"id": ["M0002", "M0003"], "msg_norm": ["b", "c"], "split": ["test", "test"]}).to_parquet(
        tmp_path / label_set.PHRASE_FILE)
    human = label_set.answer_cells({"reason": None, "lines": [line("외과", "외과")]})
    test = [["M0002", "외과", "문구2", *human, None], ["M0003", "외과", "문구3", *[None] * (len(label_set.TEST_HEADERS) - 3)]]
    cells = label_set.answer_cells(ok)
    train = [[None, None, label_set.ORDER[3], "M0001", "문구", *cells, None, "외과", 1, 0.0, "요약", *cells]]
    label_set.write_workbook(tmp_path / label_set.SHEET_FILE, train, test, 3)
    with (tmp_path / label_set.PROVISIONAL_FILE).open("w", encoding="utf-8") as f:
        for pid in ("M0002", "M0003"):
            f.write(json.dumps({"id": pid, **ok}, ensure_ascii=False) + "\n")
    prov = label_set.load_provisional_answers(tmp_path)
    assert set(prov) == {"c"} and prov["c"]["provisional"]                      # 사람이 채운 M0002 는 빠짐
    merged = message_answers.load(tmp_path)
    assert merged["b"]["source"] == "label" and merged["c"]["source"] == "provisional"
    assert set(message_answers.load(tmp_path, provisional=False)) == {"b"}
