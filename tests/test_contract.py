"""계약 테스트. 알고리즘 구현을 바꾼 뒤에도 이 테스트는 통과해야 한다."""
from fastapi.testclient import TestClient

import algorithm
from contracts.schemas import RecommendationResult, TriageResult
from server.main import app

client = TestClient(app)


# ---------- 규칙 ②: 알고리즘 함수 ----------

def test_analyze_symptoms_returns_contract_type():
    result = algorithm.analyze_symptoms("아버지가 갑자기 가슴이 아프다고 쓰러지셨어요", 68, [])
    assert isinstance(result, TriageResult)
    if result.status == "complete":
        assert result.ktas is not None
    else:
        assert result.follow_up_question


def test_rank_hospitals_sorted_and_limited():
    result = algorithm.rank_hospitals(37.4979, 127.0276, 2, ["순환기내과"], ["AMI"], limit=2)
    assert isinstance(result, RecommendationResult)
    assert len(result.hospitals) <= 2
    assert [h.rank for h in result.hospitals] == sorted(h.rank for h in result.hospitals)
    if result.mode == "fallback":
        assert result.warning
        assert all(h.accept_prob is None for h in result.hospitals)


# ---------- 규칙 ①: HTTP API ----------

def test_triage_follow_up_flow():
    r1 = client.post("/v1/triage", json={"text": "아파요"})
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["status"] == "need_more_info"
    assert body1["follow_up_question"]

    r2 = client.post("/v1/triage", json={"text": "가슴이 쥐어짜듯 아프대요", "session_id": body1["session_id"]})
    assert r2.status_code == 200
    assert r2.json()["status"] == "complete"
    assert r2.json()["ktas"] == 2


def test_triage_unknown_session_returns_error_format():
    r = client.post("/v1/triage", json={"text": "가슴이 아파요 많이요", "session_id": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_recommendations_shape():
    r = client.post("/v1/recommendations", json={"lat": 37.4979, "lon": 127.0276, "ktas": 2,
                                                 "departments": ["순환기내과"], "limit": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] in ("ai", "fallback")
    assert body["generated_at"].endswith("+09:00")
    for h in body["hospitals"]:
        assert {"rank", "hpid", "name", "eta_min", "accept_prob", "data_age_min", "reason"} <= h.keys()


def test_validation_error_format():
    r = client.post("/v1/recommendations", json={"lat": 999, "lon": 127.0, "ktas": 2})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_hospital_detail_and_not_found():
    assert client.get("/v1/hospitals/A1100010").status_code == 200
    r = client.get("/v1/hospitals/XXXX")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "HOSPITAL_NOT_FOUND"
