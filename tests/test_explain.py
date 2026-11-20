"""추천 사유 문장(explain) 테스트."""
from algorithm import explain as ex
from contracts.schemas import RankedHospital


def _hospital(**kw):
    base = dict(rank=1, hpid="A1", name="A병원", lat=37.0, lon=127.0, distance_km=3.24, eta_min=8,
                accept_prob=None, beds_available=3, data_age_min=4, notices=[])
    base.update(kw)
    return RankedHospital(**base)


def test_basic_facts():
    s = ex.explain(_hospital(), 3, ["외과"], [])
    assert "예상 도착 8분(3.2km)" in s
    assert "가용 병상 3개" in s
    assert "최근 갱신" in s


def test_shap_top_factors():
    s = ex.explain(_hospital(accept_prob=0.88), 3, ["외과"], [],
                   {"beds_available": 0.8, "data_age_min": 0.3, "hour": 0.1, "beds_trend": -0.4})
    assert "수용 확률 88%" in s
    assert "높인 요인: 응급실 가용 병상, 정보 갱신 시각" in s
    assert "낮춘 요인: 최근 병상 변화" in s
    assert "시간대" not in s          # 상위 2개만


def test_relevant_notice_warning():
    h = _hospital(notices=["[심근경색의 재관류중재술] 응급검사 및 시술 필요한 환자 수용 불가", "피부과 진료 불가"])
    s = ex.explain(h, 2, ["순환기내과"], ["Y0010"])
    assert "주의: '[심근경색의 재관류중재술]" in s
    assert "피부과" not in s
    assert "꼭 확인" in s


def test_stale_and_overcrowded():
    s = ex.explain(_hospital(beds_available=-2, data_age_min=95), 4, ["정형외과"], [])
    assert "가용 병상 없음(과밀)" in s
    assert "95분 전" in s


def test_llm_polish_rejects_changed_numbers(monkeypatch):
    monkeypatch.setenv("EXPLAIN_LLM", "1")

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "약 10분이면 도착합니다."}}
    monkeypatch.setattr(ex.requests, "post", lambda *a, **k: Resp())
    s = ex.explain(_hospital(), 3, ["외과"], [])
    assert s.startswith("예상 도착 8분")      # 숫자가 바뀐 LLM 문장은 버림
