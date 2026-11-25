import pandas as pd

from algorithm.message_match import Patient
from forecast import simulate

NOW = pd.Timestamp("2026-11-16 10:00", tz="Asia/Seoul")


class FakeWorld:
    """A(가장 가까움, 만실), B(자리 있음, 신경과 불가 공지), C(조금 멂, 자리 있지만 만실 위험 높음), D(멂, 안전)."""

    def __init__(self):
        self.hospitals = pd.DataFrame({"lat": [37.500, 37.505, 37.510, 37.530], "lon": [127.0] * 4},
                                      index=pd.Index(list("ABCD"), name="hpid"))
        self.beds_now = {"A": 0, "B": 5, "C": 2, "D": 8}
        self.risk = {"A": 0.9, "B": 0.01, "C": 0.8, "D": 0.02}

    def beds(self, h, at):
        return float(self.beds_now[h])

    def p_full(self, h, at, eta):
        return self.risk[h]

    def msg_status(self, h, posted_at, patient, at):
        return "불가" if h == "B" and "신경과" in patient.departments else "가능"


STROKE = Patient.of(["신경과"], ["Y0020"])
FRACTURE = Patient.of(["정형외과"])


def pick(policy, patient, penalty=30):
    return simulate.choose(FakeWorld(), policy, patient, 37.5, 127.0, NOW, set(), penalty)[0]


def test_baselines():
    assert pick("nearest", STROKE) == "A"
    assert pick("nearest_beds_now", STROKE) == "B"          # 공지를 보지 않음
    assert pick("nearest_beds_msg", STROKE) == "C"          # 신경과 불가 공지 제외


def test_proposed_avoids_message_block_and_full_risk():
    assert pick("proposed", STROKE) == "D"                  # B 는 공지, C 는 만실 위험
    assert pick("proposed", FRACTURE) == "B"                # 공지가 해당 없는 환자는 B
    assert pick("risk_only", STROKE) == "B"                 # 공지를 모르면 B


def test_low_penalty_prefers_closer():
    assert pick("proposed", STROKE, penalty=0) == "A"
