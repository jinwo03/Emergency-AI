import json

from preprocessing import label_set, parse_local


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self.content}}


ITEM = {"id": "M0001", "msg_norm": "신경외과 수술 불가", "msg": "신경외과 수술 불가", "dept": "신경외과"}
GOOD = {"reason": "", "lines": [{"target": "신경외과 수술", "departments": ["신경외과"], "disease_codes": [],
                                 "conditions": [], "time": "항상", "status": "불가", "caution": ""}],
        "memo": "", "confidence": 0.9}


def test_schema_reason_is_plain_enum():
    reason = parse_local.ollama_schema()["properties"]["reason"]
    assert reason["type"] == "string" and "" in reason["enum"] and set(label_set.REASONS) <= set(reason["enum"])
    assert "anyOf" in parse_local.OUTPUT_SCHEMA["properties"]["reason"]   # 원본은 그대로


def test_call_sends_prompt_and_schema():
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, body=json)
        return FakeResponse("{}")

    parse_local.call(ITEM, "qwen2.5:7b", "http://x", post=post)
    assert sent["url"] == "http://x/api/chat"
    assert sent["body"]["messages"][0]["content"] == parse_local.SYSTEM_PROMPT
    assert "<message>신경외과 수술 불가</message>" in sent["body"]["messages"][1]["content"]
    assert sent["body"]["options"]["temperature"] == 0


def test_record_matches_parse_format():
    r = parse_local.to_record(ITEM, json.dumps(GOOD, ensure_ascii=False), "qwen2.5:7b", 1.0)
    assert r["reason"] is None and r["errors"] == []
    assert r["lines"] == [["신경외과 수술", "신경외과", "", "", "항상", "불가", ""]]
    assert r["prompt_version"] == parse_local.PROMPT_VERSION


def test_broken_json_is_saved_as_error():
    r = parse_local.to_record(ITEM, "{not json", "qwen2.5:7b", 1.0)
    assert r["errors"] == ["json"] and r["lines"] == []


def test_resume_skips_same_model_and_prompt(tmp_path):
    path = parse_local.out_path(tmp_path, "qwen2.5:7b")
    assert path.name == "local_parse_qwen2.5_7b.jsonl"
    rec = parse_local.to_record(ITEM, json.dumps(GOOD, ensure_ascii=False), "qwen2.5:7b", 1.0)
    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert parse_local.done_keys(path, "qwen2.5:7b") == {ITEM["msg_norm"]}
    assert parse_local.done_keys(path, "other") == set()
