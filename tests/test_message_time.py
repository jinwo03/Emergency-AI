import pytest

from preprocessing import message_time, time_rule


@pytest.mark.parametrize("text, status, expected", [
    ("평일 야간 및 주말 진료 불가합니다.", "불가", [("평일 야간, 주말", "불가")]),
    ("야간, 주말, 공휴일은 산부인과 진료 불가함.", "불가", [("야간, 휴일", "불가")]),
    ("월,수,금 야간진료(18:00~08:00) 의료진 부재로 진료불가", "불가", [("월·수·금 18:00-08:00", "불가")]),
    ("PTBD 가능 시간: 평일 8시 ~ 17시", "불가", [("평일 08:00-17:00", "가능"), ("그 외", "불가")]),
    ("[치과] 정규시간 외 불가", "불가", [("평일 주간", "가능"), ("그 외", "불가")]),
    ("8AM ~ 8PM 환자 수용 가능", "불가", [("08:00-20:00", "가능"), ("그 외", "불가")]),
    ("[내과] 23시 이후 진료 불가", "불가", [("23:00-08:00", "불가")]),
    ("이비인후과 야간 진료 원활하지 않습니다. 반드시 연락주세요.", "어려움", [("야간", "어려움")]),
    ("[성형외과] 월요일, 목요일 정규시간(09:00~18:00) 진료가능", "불가",
     [("월·목 09:00-18:00", "가능"), ("그 외", "불가")]),
])
def test_extract(text, status, expected):
    assert message_time.extract(text, status) == expected


@pytest.mark.parametrize("text, status", [
    ("신경외과 수술 불가", "불가"),                      # 시간 없음
    ("평일 야간 진료 가능", "가능"),                     # 제한이 없으면 시간 조건을 붙이지 않음
    ("학회 일정으로 10.30 ~ 10.31 휴진", "불가"),         # 날짜는 시각으로 보지 않음
    ("24시간 진료 가능하나 소아 불가", "불가"),
])
def test_no_time(text, status):
    assert message_time.extract(text, status) is None


def test_existing_only_stays_limit_form():
    out = message_time.extract("평일(18시~익일 06시) 본원 f/u 환자만 수용 가능합니다", "기존 환자만")
    assert out == [("평일 18:00-06:00", "기존 환자만")]


def test_every_result_parses():
    for text in ["매일 0AM~7AM 단순봉합 불가", "토요일 오후13시 ~ 월요일 오전8시 불가", "(월-목)오후 9시까지 가능"]:
        out = message_time.extract(text, "불가")
        for spec, _ in out or []:
            time_rule.parse(spec)
