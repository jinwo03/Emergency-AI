"""증상 분석(triage) v2 후처리·안전장치 테스트. LLM 응답은 가짜(monkeypatch)로 대체한다."""
import pytest

from algorithm import triage
from contracts.schemas import TriageStatus


def _fake_llm(response: dict):
    """정해진 응답을 돌려주는 가짜 LLM"""
    return lambda text, age, history: response


def _llm(ktas=3, departments=(), conditions=()):
    return {"ktas": ktas, "departments": list(departments), "conditions": list(conditions)}


@pytest.fixture
def llm_backend(monkeypatch):
    monkeypatch.setenv("TRIAGE_BACKEND", "llm")


def test_llm_result_is_normalized(llm_backend, monkeypatch):
    # 허용 목록 밖 진료과·질환 코드는 제거되고, 별칭은 E-Gen 표기로 바뀐다
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(3, ["심장내과", "없는과", "심장내과"], ["y0010", "AMI"])))
    r = triage.analyze_symptoms("아버지가 계속 속이 안 좋다고 하세요", 68, [])
    assert r.status == TriageStatus.complete
    assert r.departments == ["순환기내과"]
    assert r.conditions == ["Y0010"]


def test_red_flag_overrides_under_triage(llm_backend, monkeypatch):
    # LLM이 KTAS 5로 낮게 봐도 마비·말 어눌함이 있으면 2 이하로 강제
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(5, ["신경과"], ["Y0020"])))
    r = triage.analyze_symptoms("엄마가 갑자기 말이 어눌해지고 오른쪽 팔에 힘이 없어요", 70, [])
    assert r.ktas == 2


def test_condition_code_sets_ceiling(llm_backend, monkeypatch):
    # 규칙엔 안 걸려도 LLM이 위급 질환 코드를 냈으면 그 질환의 상한 적용
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(4, ["안과"], ["Y0160"])))
    r = triage.analyze_symptoms("공사 현장에서 뭔가 튀어서 다쳤어요 계속 아프대요", 40, [])
    assert r.ktas == 2


def test_rule_adds_missed_condition(llm_backend, monkeypatch):
    # LLM이 질환과 KTAS를 모두 놓쳐도 화상 규칙이 코드와 상한을 보충
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(5, ["외과"], [])))
    r = triage.analyze_symptoms("아이가 끓는 물을 뒤집어써서 배가 다 데었어요", 4, [])
    assert r.ktas == 2
    assert r.conditions == ["Y0120"]
    assert "소아청소년과" in r.departments


def test_negation_and_chronic_do_not_trigger(llm_backend, monkeypatch):
    # "안 아프고", "한 달 됐어요" 같은 표현에서는 규칙이 KTAS를 올리지 않음
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(5, ["내과"], [])))
    assert triage.analyze_symptoms("가슴은 안 아프고 그냥 감기 기운이에요", 30, []).ktas == 5
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(4, ["신경과"], ["Y0020"])))
    assert triage.analyze_symptoms("할머니 팔에 힘이 없으신 지 한 달 됐어요", 79, []).ktas == 4


def test_age_plausibility(llm_backend, monkeypatch):
    # 성인에게 소아청소년과·영유아 코드는 빠지고, 영유아 코드 쌍은 나이에 맞게 바뀐다
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(2, ["소아청소년과", "내과"], ["Y0092"])))
    r = triage.analyze_symptoms("아버지가 기침하면서 피가 반 컵 정도 나왔어요", 66, [])
    assert "소아청소년과" not in r.departments
    assert r.conditions == ["Y0091"]
    assert r.departments[0] == "호흡기내과"


def test_symptomatic_text_is_not_delayed(llm_backend, monkeypatch):
    # 증상을 말했으면 되묻지 않고 바로 분류 (v1은 위급 환자에게 되묻는 문제가 있었음)
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(3, ["신경과"], [])))
    r = triage.analyze_symptoms("할머니 얼굴 한쪽이 처지고 침을 흘리세요 아침엔 괜찮으셨어요", 81, [])
    assert r.status == TriageStatus.complete
    assert r.ktas == 2


def test_vague_text_asks_then_classifies(llm_backend, monkeypatch):
    # 짧고 증상이 없으면 되묻고, 되묻기는 최대 2번
    monkeypatch.setattr(triage, "_call_llm", _fake_llm(_llm(3, [], [])))
    first = triage.analyze_symptoms("할머니가 좀 이상하세요", 80, [])
    assert first.status == TriageStatus.need_more_info
    assert first.follow_up_question
    final = triage.analyze_symptoms("모르겠어요", 80, ["할머니가 좀 이상하세요", "그냥 이상해요"])
    assert final.status == TriageStatus.complete
    assert final.departments == ["응급의학과"]


def test_llm_failure_falls_back_to_rules(llm_backend, monkeypatch):
    # LLM 연결 실패 시 규칙 기반 결과로 대체 (서버가 죽지 않음)
    def broken(text, age, history):
        raise ConnectionError("ollama 꺼짐")
    monkeypatch.setattr(triage, "_call_llm", broken)
    r = triage.analyze_symptoms("아버지가 가슴이 쥐어짜듯 아프대요", 68, [])
    assert r.status == TriageStatus.complete
    assert r.ktas == 2
    assert r.conditions == ["Y0010"]
    assert r.departments[0] == "순환기내과"
    assert triage.LAST_SOURCE == "rule"


def test_short_text_with_red_flag_is_not_delayed():
    # 짧아도 위험 신호가 있으면 되묻지 않음
    r = triage.analyze_symptoms("숨을 안 쉬어요", None, [])
    assert r.status == TriageStatus.complete
    assert r.ktas == 1
