import json
import sqlite3

import pandas as pd
import pytest

from egen import store
from preprocessing import clean, grid, run


def seen(hour: int, minute: int) -> str:
    return f"2026-11-01T{hour:02d}:{minute:02d}:00+09:00"


def egen(hour: int, minute: int) -> str:
    return f"20261101{hour:02d}{minute:02d}00"


def bed(hvec, hvidate, **extra) -> dict:
    return {"hpid": "A1", "hvidate": hvidate, "hvec": hvec, "hvs01": "10", **extra}


def message(msg, code="Y000", kind="응급", start=None) -> dict:
    return {"hpid": "A1", "symBlkMsgTyp": kind, "symTypCod": code, "symBlkMsg": msg,
            "symBlkSttDtm": start or egen(1, 0), "symBlkEndDtm": egen(23, 0)}


@pytest.fixture
def db():
    """샘플 데이터를 지우고 시나리오 전용 데이터만 남긴다."""
    with store.connect() as conn:
        for table in ("hospitals", "bed_snapshots", "messages", "collection_runs"):
            conn.execute(f"DELETE FROM {table}")
        store.save_hospitals(conn, [
            {"hpid": "A1", "dutyName": "병원1", "wgs84Lat": "37.5", "wgs84Lon": "127.0"},
            {"hpid": "A9", "dutyName": "병상 없는 기관", "wgs84Lat": "37.6", "wgs84Lon": "127.1"},
        ], seen(0, 0))
    return store.db_path()


def save_beds(items, seen_at) -> None:
    with store.connect() as conn:
        store.save_bed_snapshots(conn, items, seen_at)


def cleaned_beds() -> pd.DataFrame:
    with clean.open_readonly() as conn:
        return clean.clean_beds(conn)[0]


def build(run_times) -> pd.DataFrame:
    with store.connect() as conn:
        for t in run_times:
            store.log_run(conn, t, "beds", ok=True)
    with clean.open_readonly() as conn:
        beds, _ = clean.clean_beds(conn)
        runs = clean.load_bed_runs(conn)
    return grid.build_grid(beds, runs)[0]


def test_readonly_connection_refuses_writes(db):
    with clean.open_readonly() as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM hospitals")


def test_run_leaves_source_rows_unchanged(db, tmp_path):
    save_beds([bed("5", egen(1, 0))], seen(1, 0))
    with store.connect() as conn:
        store.log_run(conn, seen(1, 0), "beds", ok=True)
        before = [tuple(r) for r in conn.execute("SELECT * FROM bed_snapshots")]

    report = run.run(tmp_path)

    with store.connect() as conn:
        assert [tuple(r) for r in conn.execute("SELECT * FROM bed_snapshots")] == before
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["grid"] == report["grid"]
    assert (tmp_path / "beds_grid.parquet").exists()


def test_clean_beds_keeps_negative_and_drops_null(db):
    save_beds([bed("-3", egen(1, 0)), bed("", egen(1, 5))], seen(1, 5))
    with clean.open_readonly() as conn:
        beds, stats = clean.clean_beds(conn)
    assert beds["hvec"].tolist() == [-3]
    assert stats["null_hvec_dropped"] == 1


def test_clean_beds_unreported_field_is_missing_not_zero(db):
    save_beds([bed("3", egen(1, 0), hvoc="2"), bed("4", egen(1, 5))], seen(1, 5))
    beds = cleaned_beds()
    assert beds["hvoc_reported"].tolist() == [True, False]
    assert beds["hvoc"].iloc[0] == 2 and pd.isna(beds["hvoc"].iloc[1])


def test_clean_beds_keeps_equipment_status(db):
    save_beds([
        bed("3", egen(1, 0), hvctayn="Y"),
        bed("3", egen(1, 5), hvctayn="N"),
        bed("3", egen(1, 10), hvctayn="N1"),
        bed("3", egen(1, 15)),
    ], seen(1, 15))
    beds = cleaned_beds()
    assert beds["ct_status"].tolist()[:3] == ["Y", "N", "N1"]
    assert beds["ct_available"].tolist()[:3] == [True, False, False]
    assert pd.isna(beds["ct_available"].iloc[3])


def test_clean_beds_flags_suspect_values_without_changing_them(db):
    save_beds([bed("11", egen(1, 0)), bed("-11", egen(1, 5)), bed("-10", egen(1, 10))], seen(1, 10))
    beds = cleaned_beds()
    assert beds["hvec"].tolist() == [11, -11, -10]
    assert beds["hvec_suspect"].tolist() == [True, True, False]


def test_clean_beds_drops_older_record_seen_later(db):
    save_beds([bed("5", egen(2, 0))], seen(2, 0))
    save_beds([bed("9", egen(1, 0))], seen(2, 5))
    with clean.open_readonly() as conn:
        beds, stats = clean.clean_beds(conn)
    assert beds["hvec"].tolist() == [5]
    assert stats["out_of_order_dropped"] == 1


def test_clean_messages_drops_only_general_placeholder(db):
    with store.connect() as conn:
        store.save_messages(conn, [
            message(clean.PLACEHOLDER_MSG),
            message(clean.PLACEHOLDER_MSG, code="Y0032", kind="중증"),
            message("의료진 부족"),
        ], seen(1, 0))
    with clean.open_readonly() as conn:
        messages, stats = clean.clean_messages(conn)
    assert stats["placeholder_dropped"] == 1
    assert stats["placeholder_kept_with_disease_code"] == 1
    assert sorted(messages["sym_code"]) == ["Y000", "Y0032"]


def test_normalize_message_only_removes_spacing_case_and_trailing_period():
    assert clean.normalize_message("NS 수술  불가.") == clean.normalize_message("ns수술 불가")
    assert clean.normalize_message("수술 불가") != clean.normalize_message("수술 가능")


def test_clean_messages_numbers_repeated_registrations(db):
    with store.connect() as conn:
        store.save_messages(conn, [message("야간 불가", start=egen(1, 0))], seen(1, 0))
        store.save_messages(conn, [message("야간 불가", start=egen(2, 0))], seen(2, 0))
    with clean.open_readonly() as conn:
        messages, stats = clean.clean_messages(conn)
    assert messages["series_id"].nunique() == 1
    assert messages["episode_no"].tolist() == [1, 2]
    assert stats["repeated_series"] == 1


def test_clean_messages_marks_unknown_duration(db):
    with store.connect() as conn:
        store.save_messages(conn, [message("처음부터 있음")], seen(1, 0))
        store.save_messages(conn, [message("중간에 사라짐", start=egen(1, 30))], seen(1, 30))
        store.save_messages(conn, [message("지금도 있음", start=egen(2, 0))], seen(2, 0))
    with clean.open_readonly() as conn:
        messages, _ = clean.clean_messages(conn)
    flags = messages.set_index("msg")
    assert flags.loc["처음부터 있음", "left_censored"]
    assert flags.loc["지금도 있음", "right_censored"]
    assert not flags.loc["중간에 사라짐", "left_censored"]
    assert not flags.loc["중간에 사라짐", "right_censored"]


def test_clean_hospitals_drops_those_without_beds(db):
    save_beds([bed("5", egen(1, 0))], seen(1, 0))
    with clean.open_readonly() as conn:
        beds, _ = clean.clean_beds(conn)
        hospitals, stats = clean.clean_hospitals(conn, beds["hpid"])
    assert hospitals["hpid"].tolist() == ["A1"]
    assert stats["without_bed_data_dropped"] == 1


def test_grid_uses_only_what_was_seen_by_t(db):
    # 01:04 에 갱신된 값이지만 01:10 에야 수집됨 → 01:05 시점에는 알 수 없음
    save_beds([bed("5", egen(1, 0))], seen(1, 0))
    save_beds([bed("9", egen(1, 4))], seen(1, 10))
    result = build([seen(1, 0), seen(1, 5), seen(1, 10)])
    assert result["hvec"].tolist() == [5, 5, 9]


def test_grid_marks_values_older_than_30_minutes_stale(db):
    save_beds([bed("5", egen(1, 0))], seen(1, 0))
    result = build([seen(1, m) for m in range(0, 40, 5)]).set_index("data_age_min")
    assert result.loc[30, "hvec"] == 5 and not result.loc[30, "is_stale"]
    assert pd.isna(result.loc[35, "hvec"]) and result.loc[35, "is_stale"]


def test_grid_clips_hospital_clock_ahead_of_collector(db):
    save_beds([bed("5", egen(1, 3))], seen(1, 0))
    assert build([seen(1, 0)])["data_age_min"].tolist() == [0]


def test_grid_flags_collection_gap(db):
    save_beds([bed("5", egen(1, 0))], seen(1, 0))
    result = build([seen(1, 0), seen(1, 15)])
    assert result["collection_gap"].tolist() == [False, True, True, False]


def test_grid_starts_each_hospital_when_first_seen(db):
    save_beds([bed("5", egen(1, 0))], seen(1, 0))
    save_beds([{"hpid": "A2", "hvidate": egen(1, 5), "hvec": "1"}], seen(1, 5))
    result = build([seen(1, 0), seen(1, 5)])
    assert (result["hpid"] == "A1").sum() == 2
    assert (result["hpid"] == "A2").sum() == 1


def test_hospital_quality_only_flags(db):
    save_beds([bed("5", egen(1, 0))], seen(1, 0))
    save_beds([bed("5", egen(1, 5))], seen(1, 5))
    result = build([seen(1, 0), seen(1, 5)])
    with clean.open_readonly() as conn:
        beds, _ = clean.clean_beds(conn)
        hospitals, _ = clean.clean_hospitals(conn, beds["hpid"])
    quality, stats = grid.hospital_quality(hospitals, beds, result)
    assert quality["hpid"].tolist() == ["A1"]
    assert quality["constant_hvec"].iloc[0] and quality["exclude_candidate"].iloc[0]
    assert stats["constant_hvec"] == 1
