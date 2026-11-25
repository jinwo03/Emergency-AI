import json
from types import SimpleNamespace

import pandas as pd
import pytest

from egen import store
from preprocessing import clean, label_set, message_answers, message_status, parse_messages, run
from preprocessing.message_status import AVAILABLE, BLOCKED, DIFFICULT, EXISTING_ONLY, UNKNOWN, UNPARSED


def seen(hour: int, minute: int = 0) -> str:
    return f"2026-11-01T{hour:02d}:{minute:02d}:00+09:00"


def egen(hour: int, minute: int = 0) -> str:
    return f"20261101{hour:02d}{minute:02d}00"


def message(msg, code="Y000", kind="응급", dept="외과", end=None) -> dict:
    return {"hpid": "A1", "symBlkMsgTyp": kind, "symTypCod": code, "trtPrtCodMag": dept, "symBlkMsg": msg,
            "symBlkSttDtm": egen(1), "symBlkEndDtm": end or egen(23)}


@pytest.fixture
def db():
    with store.connect() as conn:
        for table in ("hospitals", "bed_snapshots", "messages", "collection_runs"):
            conn.execute(f"DELETE FROM {table}")
        store.save_hospitals(conn, [{"hpid": "A1", "dutyName": "병원1", "wgs84Lat": "37.5", "wgs84Lon": "127.0"}],
                             seen(0))
        store.save_bed_snapshots(conn, [{"hpid": "A1", "hvidate": egen(1), "hvec": "5"}], seen(1))
        store.log_run(conn, seen(1), "beds", ok=True)
        store.log_run(conn, seen(1, 10), "beds", ok=True)


def save_messages(items, seen_at) -> None:
    with store.connect() as conn:
        store.save_messages(conn, items, seen_at)


def cleaned() -> pd.DataFrame:
    with clean.open_readonly() as conn:
        return clean.clean_messages(conn)[0]


def line(target="외과", dept="외과", codes="", conds="", time="항상", status=BLOCKED, caution="") -> list:
    return [target, dept, codes, conds, time, status, caution]


def parse_record(msg, lines, confidence=0.9, errors=()) -> dict:
    return {"msg_norm": clean.normalize_message(msg), "msg": msg, "reason": None, "lines": lines, "memo": "",
            "confidence": confidence, "errors": list(errors)}


def llm(msg, lines, **kw) -> dict:
    return {clean.normalize_message(msg): {"source": "llm", **parse_record(msg, lines, **kw)}}


def api_line(status=BLOCKED, departments=("외과",), time="항상") -> dict:
    return {"target": "외과", "departments": list(departments), "disease_codes": [], "conditions": [],
            "time": time, "status": status, "caution": ""}


@pytest.mark.parametrize("text, expected", [
    ("1234567이O수 시술 중.", "[환자정보] 시술 중."),
    ("12345678 박O영 IAT 시술 진행", "[환자정보] IAT 시술 진행"),
    ("수용 불가 (pf.홍길동)", "수용 불가 ([의료진])"),
    ("부인과 환자는 수용 가능 (Dr.홍길동)", "부인과 환자는 수용 가능 ([의료진])"),
    ("Dr 홍길동 부재", "[의료진] 부재"),
    ("핫라인:010-1234-5678", "핫라인:[전화번호]"),
    ("평일 8-22시 진료 가능", "평일 8-22시 진료 가능"),
    ("담당과장 부재", "담당과장 부재"),
    ("Y0032 불가", "Y0032 불가"),
])
def test_mask_personal_info(text, expected):
    assert clean.mask_personal_info(text) == expected


def test_clean_messages_masks_only_output_not_source(db):
    raw = "1234567이O수 시술 중."
    save_messages([message(raw)], seen(1))
    assert cleaned()["msg"].tolist() == ["[환자정보] 시술 중."]
    with store.connect() as conn:
        assert conn.execute("SELECT msg FROM messages").fetchone()[0] == raw


@pytest.mark.parametrize("msg, placeholder, expected", [
    ("불가능메세지등록", True, BLOCKED),
    ("의료진 부족", False, BLOCKED),
    ("시술불가", False, BLOCKED),
    ("수용 불가, 사전 문의 바람", False, BLOCKED),
    ("외상성 뇌출혈 수용 가능", False, DIFFICULT),
    ("CV 부재로 이송 전 문의 부탁드립니다", False, DIFFICULT),
    ("IAT 진행중", False, BLOCKED),
])
def test_severe_status(msg, placeholder, expected):
    assert message_status.severe_status(msg, placeholder) == expected


@pytest.mark.parametrize("msg, expected", [
    ("성형외과 진료 불가", BLOCKED),
    ("일-금 8-22시 진료 가능", DIFFICULT),
    ("인력 부족", BLOCKED),
    ("이송 및 전원에 참고 바랍니다", UNPARSED),
])
def test_general_status(msg, expected):
    assert message_status.general_status(msg) == expected


def test_targets_use_rule_without_answer(db):
    save_messages([message("시술불가", code="Y0010", kind="중증", dept=""), message("외과 진료 불가")], seen(1))
    targets, stats = message_status.build_targets(cleaned(), {})
    by_source = targets.set_index("status_source")
    assert by_source.loc["code_rule", "disease_codes"] == "Y0010"
    assert by_source.loc["code_rule", "departments"] == "전체"
    assert by_source.loc["text_rule", "departments"] == "외과"
    assert set(targets["time"]) == {"항상"}
    assert stats["general_messages_answered"] == 0


def test_rule_difficult_asks_first(db):
    save_messages([message("이송 전 문의 바랍니다")], seen(1))
    targets, _ = message_status.build_targets(cleaned(), {})
    assert targets[["status", "caution"]].values.tolist() == [[DIFFICULT, "문의"]]


def test_targets_keep_answer_lines_in_order(db):
    text = "두경부 파트 진료 불가, 그 외 평일 주간 가능"
    save_messages([message(text, dept="이비인후과")], seen(1))
    answers = llm(text, [line("두경부", "이비인후과"), line("그 외", "이비인후과", time="평일 주간", status=AVAILABLE)])
    targets, _ = message_status.build_targets(cleaned(), answers)
    assert targets[["line_no", "target", "time", "status"]].values.tolist() == [
        [1, "두경부", "항상", BLOCKED], [2, "그 외", "평일 주간", AVAILABLE]]
    assert set(targets["status_source"]) == {"llm"}


def test_targets_keep_codes_and_conditions_as_cells(db):
    text = "NS, 소아 뇌출혈 수술 불가"
    save_messages([message(text, dept="신경외과")], seen(1))
    answers = llm(text, [line("소아 뇌출혈 수술", "신경외과", "Y0031, Y0032", "소아")])
    row = message_status.build_targets(cleaned(), answers)[0].iloc[0]
    assert (row["disease_codes"], row["conditions"]) == ("Y0031, Y0032", "소아")


@pytest.mark.parametrize("kw", [{"confidence": 0.3}, {"errors": ["1줄 시간 '밤'"]}])
def test_untrusted_llm_becomes_unparsed_not_blocked(db, kw):
    text = "알 수 없는 문구"
    save_messages([message(text)], seen(1))
    targets, _ = message_status.build_targets(cleaned(), llm(text, [line()], **kw))
    assert targets["status"].tolist() == [UNPARSED]


def test_label_answer_used_even_without_confidence(db):
    text = "신환 수용 불가"
    save_messages([message(text)], seen(1))
    answers = {clean.normalize_message(text): {"source": "label", "reason": None,
                                                "lines": [line(status=EXISTING_ONLY)]}}
    targets, _ = message_status.build_targets(cleaned(), answers)
    assert targets[["status", "status_source"]].values.tolist() == [[EXISTING_ONLY, "label"]]


def test_statuses_shared_with_label_sheet():
    assert label_set.STATUSES == message_status.STATUSES
    schema_line = parse_messages.OUTPUT_SCHEMA["properties"]["lines"]["items"]["properties"]
    assert schema_line["status"]["enum"] == message_status.STATUSES
    assert UNKNOWN in schema_line["status"]["enum"]


def grid_times(*minutes) -> pd.Series:
    return pd.Series(pd.to_datetime([seen(1, m) for m in minutes])).dt.as_unit("ns")


def test_message_grid_covers_seen_interval_only(db):
    item = message("외과 진료 불가")
    save_messages([item], seen(1, 5))
    save_messages([item], seen(1, 10))
    targets, _ = message_status.build_targets(cleaned(), {})
    result, _ = message_status.build_message_grid(targets, grid_times(0, 5, 10, 15))
    assert result["t"].tolist() == grid_times(5, 10).tolist()


def test_message_grid_stops_at_published_end(db):
    item = message("외과 진료 불가", end=egen(1, 7))
    save_messages([item], seen(1))
    save_messages([item], seen(1, 15))
    targets, _ = message_status.build_targets(cleaned(), {})
    result, _ = message_status.build_message_grid(targets, grid_times(0, 5, 10, 15))
    assert result["t"].tolist() == grid_times(0, 5).tolist()


def test_message_grid_handles_no_matching_slot(db):
    save_messages([message("외과 진료 불가")], seen(1, 3))
    targets, _ = message_status.build_targets(cleaned(), {})
    result, stats = message_status.build_message_grid(targets, grid_times(0, 5))
    assert result.empty
    assert stats["targets_without_slot"] == 1


def write_parses(labels, records, name=message_answers.CLASSIFIER_FILE) -> None:
    labels.mkdir(parents=True, exist_ok=True)
    (labels / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def test_run_writes_message_outputs(db, tmp_path):
    save_messages([message("외과 진료 불가")], seen(1))
    report = run.run(tmp_path / "processed")
    assert report["message_targets"]["rows"] == 1
    assert (tmp_path / "processed" / "message_grid.parquet").exists()


def test_run_reads_classifier_results_from_labels_folder(db, tmp_path):
    text = "외과 진료 불가"
    save_messages([message(text)], seen(1))
    write_parses(tmp_path / "labels", [parse_record(text, [line(codes="Y0060")])])
    assert run.run(tmp_path / "processed")["message_targets"]["by_source"] == {"classifier": 1}


def test_llm_parses_are_not_used_for_recommendation(tmp_path):
    """LLM 해석은 정답 초안용이라 추천에 쓰는 답 모음에 들어가지 않는다."""
    write_parses(tmp_path, [parse_record("외과 진료 불가", [line()])], name=message_status.PARSE_FILE)
    assert message_answers.load(tmp_path) == {}


def test_low_confidence_classifier_result_is_still_usable():
    answer = {"source": "classifier", "confidence": 0.3, "errors": [], "lines": [line(status="판단 불가")]}
    assert message_status.usable(answer)
    assert not message_status.usable(answer | {"errors": ["1줄 시간 'x'"]})


def test_label_answer_overrides_classifier(tmp_path, monkeypatch):
    text = "외과 진료 불가"
    key = clean.normalize_message(text)
    write_parses(tmp_path, [parse_record(text, [line(status=AVAILABLE)])])
    monkeypatch.setattr(label_set, "load_label_answers",
                        lambda d: ({key: {"id": "M0001", "split": "train", "reason": None, "lines": [line()]}}, {}))
    answers = message_answers.load(tmp_path)
    assert answers[key]["source"] == "label"
    assert answers[key]["lines"] == [line()]


def test_answer_from_cells_round_trip():
    answer = {"reason": "의료진 부재", "lines": [line("두경부", "이비인후과"), line("그 외", "이비인후과", status=AVAILABLE)]}
    assert label_set.answer_from_cells(label_set.answer_cells(answer)) == answer
    assert label_set.answer_from_cells([None] * len(label_set.ANSWER_HEADERS)) == {"reason": None, "lines": []}


def test_find_unparsed_skips_labeled_parsed_and_severe(db, tmp_path, monkeypatch):
    save_messages([
        message("1234567이O수 시술 중."),
        message("외과 진료 불가"),
        message("외과 진료  불가."),  # 정규화하면 같은 문구
        message("소아 진료 불가"),
        message("시술불가", code="Y0010", kind="중증"),
    ], seen(1))
    monkeypatch.setattr(label_set, "load_label_answers",
                        lambda d: ({clean.normalize_message("소아 진료 불가"): {}}, {}))
    items = parse_messages.find_unparsed(tmp_path)
    assert sorted(i["msg_norm"] for i in items) == ["[환자정보]시술중", "외과진료불가"]
    assert all("이O수" not in i["msg"] for i in items)

    write_parses(tmp_path, [parse_record("외과 진료 불가", [line()], errors=["1줄 시간 '밤'"])], name=message_status.PARSE_FILE)
    assert [i["msg_norm"] for i in parse_messages.find_unparsed(tmp_path)] == ["[환자정보]시술중"]
    assert len(parse_messages.find_unparsed(tmp_path, retry_invalid=True)) == 2


def test_system_prompt_built_from_sheet_guide():
    prompt = parse_messages.SYSTEM_PROMPT
    for text in ("기존 환자만", "문의·지연", "줄 순서", "Y0020", "중환자실 필요", "흉부외과", "두경부 파트 | 이비인후과"):
        assert text in prompt
    assert "노란 칸" not in prompt and "'목록' 시트" not in prompt


def test_build_request_constrains_output_and_sends_masked_text():
    item = {"msg_norm": "x", "msg": "[환자정보] 시술 중", "dept": "외과"}
    request = parse_messages.build_request(item, "claude-opus-5-5")
    assert len(request["custom_id"]) == 32
    assert request["params"]["output_config"]["format"]["schema"] == parse_messages.OUTPUT_SCHEMA
    assert "[환자정보] 시술 중" in request["params"]["messages"][0]["content"]


def succeeded(item, payload, stop_reason="end_turn"):
    block = SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))
    message = SimpleNamespace(stop_reason=stop_reason, content=[block])
    return SimpleNamespace(custom_id=parse_messages.custom_id(item["msg_norm"]),
                           result=SimpleNamespace(type="succeeded", message=message))


class FakeBatches:
    """네트워크 없이 Batches API 흐름만 검증한다."""

    def __init__(self, results):
        self._results = results
        self.created = None

    def create(self, requests):
        self.created = requests
        return SimpleNamespace(id="batch_1")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended", request_counts=SimpleNamespace(processing=0))

    def results(self, batch_id):
        return iter(self._results)


def payload(*lines, confidence=0.9) -> dict:
    return {"reason": None, "lines": list(lines), "memo": "", "confidence": confidence}


def test_submit_then_collect_saves_answers_and_retries_failures(tmp_path):
    items = {k: {"msg_norm": k, "msg": k, "dept": "외과"} for k in ("좋음", "시간오류", "거절", "JSON오류", "오류")}
    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches([
        succeeded(items["좋음"], payload(api_line(departments=["외과", "정형외과"]))),
        succeeded(items["시간오류"], payload(api_line(time="밤"))),
        succeeded(items["거절"], payload(api_line()), stop_reason="refusal"),
        succeeded(items["JSON오류"], {"lines": [], "confidence": 7}),
        SimpleNamespace(custom_id=parse_messages.custom_id("오류"), result=SimpleNamespace(type="errored")),
    ])))
    labels = tmp_path / "labels"

    parse_messages.submit(client, list(items.values()), labels, "m")
    assert (labels / parse_messages.PENDING_FILE).exists()
    assert parse_messages.collect(client, labels) == {"saved": 2, "saved_with_errors": 1, "failed": 3}
    assert not (labels / parse_messages.PENDING_FILE).exists()

    parses = message_status.load_parses(labels / message_status.PARSE_FILE)
    assert parses["좋음"]["lines"] == [line(dept="외과, 정형외과")]
    assert parses["좋음"]["errors"] == []
    assert parses["좋음"]["prompt_version"] == parse_messages.PROMPT_VERSION
    assert parses["시간오류"]["errors"] and "시간" in parses["시간오류"]["errors"][0]


def test_export_review_writes_one_row_per_line(tmp_path):
    write_parses(tmp_path, [parse_record("A", [line(), line(status=AVAILABLE)]), parse_record("안내문", [])], name=message_status.PARSE_FILE)
    path, rows = parse_messages.export_review(tmp_path)
    assert rows == 3
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_pilot_items_only_labeled_train(tmp_path, monkeypatch):
    pd.DataFrame({"id": ["M1", "M2", "M3"], "msg_norm": ["a", "b", "c"], "msg": ["A", "B", "C"],
                  "dept": ["외과"] * 3, "split": ["train", "train", "test"]}).to_parquet(tmp_path / label_set.PHRASE_FILE)
    monkeypatch.setattr(label_set, "load_label_answers", lambda d: ({"a": {}, "c": {}}, {}))
    assert [i["msg_norm"] for i in parse_messages.pilot_items(tmp_path, 10)] == ["a"]


def test_collect_writes_to_pending_out_file(tmp_path):
    item = {"msg_norm": "x", "msg": "x", "dept": "외과"}
    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches([succeeded(item, payload(api_line()))])))
    parse_messages.submit(client, [item], tmp_path, "m", out_file=parse_messages.PILOT_FILE)
    parse_messages.collect(client, tmp_path)
    assert (tmp_path / parse_messages.PILOT_FILE).exists()
    assert not (tmp_path / message_status.PARSE_FILE).exists()
