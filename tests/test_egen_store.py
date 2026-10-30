from fastapi.testclient import TestClient

import algorithm
from egen import store
from server.main import app

client = TestClient(app)
GANGNAM = (37.4979, 127.0276)


def test_bed_snapshot_dedup_by_hvidate():
    item = {"hpid": "A1100010", "hvidate": "20261030120000", "hvec": "3"}
    with store.connect() as conn:
        assert store.save_bed_snapshots(conn, [item], store.iso(store.now_kst())) == 1
        assert store.save_bed_snapshots(conn, [item], store.iso(store.now_kst())) == 0


def test_messages_grouped_by_text_and_expired_excluded():
    with store.connect() as conn:
        notices = store.active_messages(conn, ["A1100010"])["A1100010"]
    assert notices == ["[대동맥응급(복부), 수족지접합] 의료진 부족"]


def test_baseline_prefers_beds_then_distance():
    result = algorithm.rank_hospitals(*GANGNAM, 2, [], [], limit=3)
    assert result.mode == "fallback"
    assert result.warning
    names = [h.hpid for h in result.hospitals]
    # 세브란스가 가장 가깝지만 과밀(hvec=-2) → 병상 있는 곳 뒤로
    assert names == ["A1100010", "A1100043", "A1100015"]
    first = result.hospitals[0]
    assert first.accept_prob is None
    assert 2 <= first.data_age_min <= 4
    assert first.notices


def test_no_data_returns_503():
    with store.connect() as conn:
        conn.execute("DELETE FROM bed_snapshots")
    r = client.post("/v1/recommendations", json={"lat": GANGNAM[0], "lon": GANGNAM[1], "ktas": 2})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "DATA_UNAVAILABLE"
