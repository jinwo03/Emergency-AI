"""증상 분석(triage) 후처리·안전장치 테스트. LLM 응답은 가짜(monkeypatch)로 대체한다."""
import pytest

from algorithm import triage
from contracts.schemas import TriageStatus


def _fake_llm(response: dict):
    """정해진 응답을 돌려주는 가짜 LLM"""
    return lambda text, age, history: response


@pytest.fixture
def llm_backend(monkeypatch):
    monkeypatch.setenv("TRIAGE_BACKEND", "llm")


def test_llm_result_is_normalized(llm_backend, monkeypatch):
    # 허용 목록 밖 진료과·질환 코드는 제거되고, 별칭은 E-Gen 표기로 바뀐다
    monkeypatch.setattr(triage, "_call_llm", _fake_llm({
        "need_more_info": False, "follow_up_question": "", "ktas": 3,
        "departments": ["심장내과", "없는과", "심장내과"], "conditions": ["y0010", "AMI"],
    }))
    r = triage.analyze_symptoms("아버지가 계속 속이 안 좋다고 하세요", 68, [])
    assert r.status == TriageStatus.complete
    assert r.departments == ["순환기내과"]
    assert r.conditions == ["Y0010"]


def test_red_flag_overrides_under_triage(llm_backend, monkeypatch):
    # LLM이 KTAS 5로 낮게 봐도 마비·말 어눌함이 있으면 2 이하로 강제
    monkeypatch.setattr(triage, "_call_llm", _fake_llm({
        "need_more_info": False, "follow_up_question": "", "ktas": 5,
        "departments": ["신경과"], "conditions": ["Y0020"],
    }))
    r = triage.analyze_symptoms("엄마가 갑자기 말이 어눌해지고 오른쪽 팔에 힘이 없어요", 70, [])
    assert r.ktas == 2


def test_no_follow_up_when_red_flag(llm_backend, monkeypatch):
    # 위험 신호가 있으면 LLM이 되묻자고 해도 바로 분류
    monkeypatch.setattr(triage, "_call_llm", _fake_llm({
        "need_more_info": True, "follow_up_question": "언제부터 그러셨나요?", "ktas": 2,
        "departments": ["응급의학과"], "conditions": [],
    }))
    r = triage.analyze_symptoms("아이가 갑자기 경련을 해요 어떡해요", 5, [])
    assert r.status == TriageStatus.complete


def test_follow_up_limit(llm_backend, monkeypatch):
    # 되묻기는 최대 2번까지, 그 뒤엔 무조건 분류
    monkeypatch.setattr(triage, "_call_llm", _fake_llm({
        "need_more_info": True, "follow_up_question": "어디가 아프신가요?", "ktas": 3,
        "departments": [], "conditions": [],
    }))
    first = triage.analyze_symptoms("할머니가 좀 이상하세요 걱정돼요", 80, [])
    assert first.status == TriageStatus.need_more_info
    final = triage.analyze_symptoms("잘 모르겠어요 그냥 이상해요", 80, ["할머니가 좀 이상하세요", "모르겠어요"])
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


def test_short_text_with_red_flag_is_not_delayed():
    # 짧아도 위험 신호가 있으면 되묻지 않음
    r = triage.analyze_symptoms("숨을 안 쉬어요", None, [])
    assert r.status == TriageStatus.complete
    assert r.ktas == 1
