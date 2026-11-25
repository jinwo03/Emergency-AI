from preprocessing import classify_messages as cm, label_set

ITEM = {"msg_norm": "x", "msg": "신경외과 수술 불가, 소아 진료 어려움 문의", "dept": "신경외과"}
ANSWER = {"reason": "의료진 부재", "lines": [
    ["신경외과 수술", "신경외과", "", "", "항상", "불가", ""],
    ["소아", "소아청소년과", "", "소아", "평일", "어려움", "문의·지연"],
]}


def test_encode_takes_strongest_status_and_unions():
    v = cm.encode(ANSWER)
    assert v["status"] == "불가"
    assert v["depts"] == ["소아청소년과", "신경외과"]
    assert v["conds"] == ["소아"] and v["caution"] == ["문의", "지연"] and v["reason"] == "의료진 부재"


def test_unknown_only_when_every_line_unknown():
    mixed = {"reason": None, "lines": [["a", "전체", "", "", "항상", "판단 불가", ""],
                                       ["b", "외과", "", "", "항상", "가능", ""]]}
    assert cm.encode(mixed)["status"] == "가능"
    assert cm.encode({"reason": None, "lines": [mixed["lines"][0]]})["status"] == "판단 불가"


def test_decode_is_valid_single_line():
    answer = cm.decode(cm.encode(ANSWER), ITEM)
    assert label_set.validate_answer(answer) == []
    assert len(answer["lines"]) == 1 and answer["lines"][0][4] == "항상"
    assert answer["lines"][0][6] == "문의·지연"


def test_decode_falls_back_to_message_department():
    v = {"status": "불가", "reason": cm.NO_REASON, "depts": [], "codes": [], "conds": [], "caution": []}
    assert cm.decode(v, ITEM)["lines"][0][1] == "신경외과"
    assert cm.decode(v, ITEM | {"dept": "응급실"})["lines"][0][1] == "전체"
    assert cm.decode(v | {"depts": ["전체", "외과"]}, ITEM)["lines"][0][1] == "외과"


def test_task_metrics_blocked_recall():
    gold = [cm.encode(ANSWER), {**cm.encode(ANSWER), "status": "가능"}]
    pred = [{**gold[0], "status": "어려움"}, gold[1]]
    m = cm.task_metrics(gold, pred)
    assert m["status_acc"] == 0.5 and m["status_blocked_recall"] == 0.0 and m["depts_micro_f1"] == 1.0
