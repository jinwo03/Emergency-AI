"""로컬 LLM(Ollama)으로 응급 메시지 문구를 정답 시트와 같은 줄 형식으로 바꾼다. 비용 없음.

parse_messages(Claude)와 같은 지시문·출력 형식·검사를 쓰므로 결과를 score_parses 로 바로 비교할 수 있다.

실행 (먼저 `ollama pull qwen2.5:7b` 후 Ollama 실행):
    python -m preprocessing.parse_local                        # 시험용 217종 → data/labels/local_parse_<모델>.jsonl
    python -m preprocessing.parse_local --split train --limit 50
    python -m preprocessing.parse_local --model exaone3.5:7.8b
    python -m preprocessing.score_parses --provisional --pred data/labels/local_parse_qwen2.5_7b.jsonl

- 이미 같은 모델·지시문으로 해석한 문구는 건너뛴다 (중간에 끊겨도 다시 실행하면 이어서 한다).
- JSON 이 깨지면 errors=["json"] 으로 저장한다. 채점에서는 invalid 로 센다.
- 결과 파일은 시험용 해석을 담으므로 사람이 시험용 시트를 채우기 전에는 열지 않는다.
"""
import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from pydantic import ValidationError

from egen import store
from preprocessing import label_set
from preprocessing.parse_messages import OUTPUT_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT, ParsedMessage, to_answer

DEFAULT_MODEL = os.getenv("PARSE_LOCAL_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
NUM_CTX = 8192
TIMEOUT_SEC = 300


def ollama_schema() -> dict:
    """Ollama 문법 변환기가 anyOf 를 잘 다루지 못해 reason 을 '' 포함 문자열 목록으로 바꾼다 ('' → null)."""
    schema = copy.deepcopy(OUTPUT_SCHEMA)
    schema["properties"]["reason"] = {"type": "string", "enum": ["", *label_set.REASONS]}
    return schema


def out_path(labels_dir: Path, model: str) -> Path:
    return labels_dir / f"local_parse_{re.sub(r'[^0-9A-Za-z.]+', '_', model)}.jsonl"


def load_items(labels_dir: Path, split: str) -> list[dict]:
    phrases = pd.read_parquet(labels_dir / label_set.PHRASE_FILE, columns=["id", "msg_norm", "msg", "dept", "split"])
    if split != "all":
        phrases = phrases[phrases["split"] == split]
    return [{"id": p.id, "msg_norm": p.msg_norm, "msg": p.msg, "dept": p.dept} for p in phrases.sort_values("id").itertuples()]


def done_keys(path: Path, model: str) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("model") == model and r.get("prompt_version") == PROMPT_VERSION:
                    out.add(r["msg_norm"])
    return out


def build_payload(item: dict, model: str) -> dict:
    return {
        "model": model,
        "stream": False,
        "format": ollama_schema(),
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<dept_field>{item['dept']}</dept_field>\n<message>{item['msg']}</message>"},
        ],
    }


def to_record(item: dict, text: str, model: str, seconds: float) -> dict:
    """응답 글을 parse_messages 와 같은 저장 형식으로. 형식이 깨졌으면 errors=["json"]."""
    base = {"msg_norm": item["msg_norm"], "msg": item["msg"], "model": model, "prompt_version": PROMPT_VERSION,
            "parsed_at": store.iso(store.now_kst()), "seconds": round(seconds, 2)}
    try:
        parsed = ParsedMessage.model_validate_json(text)
    except ValidationError:
        return base | {"reason": None, "lines": [], "memo": "", "confidence": 0.0, "errors": ["json"], "raw": text[:2000]}
    answer = to_answer(parsed)
    answer["reason"] = answer["reason"] or None
    return base | answer | {"confidence": parsed.confidence, "errors": label_set.validate_answer(answer)}


def call(item: dict, model: str, url: str = OLLAMA_URL, post=requests.post) -> tuple[str, float]:
    start = time.perf_counter()
    r = post(f"{url}/api/chat", json=build_payload(item, model), timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()["message"]["content"], time.perf_counter() - start


def check_server(url: str, model: str) -> str | None:
    """문제가 있으면 안내 문구, 없으면 None."""
    try:
        tags = requests.get(f"{url}/api/tags", timeout=5).json()
    except requests.RequestException:
        return f"Ollama 에 연결하지 못했습니다 ({url}). Ollama 를 실행하세요"
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        return f"모델 {model} 이 없습니다. `ollama pull {model}` 을 먼저 실행하세요 (있는 모델: {', '.join(sorted(names)) or '없음'})"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--limit", type=int, help="앞에서부터 N종만")
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--labels", type=Path, default=store.DEFAULT_DB.parent / "labels")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    problem = check_server(args.url, args.model)
    if problem:
        sys.exit(problem)
    path = out_path(args.labels, args.model)
    done = done_keys(path, args.model)
    items = [i for i in load_items(args.labels, args.split) if i["msg_norm"] not in done][:args.limit]
    print(f"{args.model}, 지시문 {PROMPT_VERSION}: 해석할 문구 {len(items)}종 (이미 {len(done)}종) → {path}", flush=True)

    counts = {"saved": 0, "json_error": 0, "format_error": 0, "failed": 0}
    total_sec = 0.0
    with path.open("a", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            try:
                text, sec = call(item, args.model, args.url)
            except (requests.RequestException, KeyError) as e:
                counts["failed"] += 1
                print(f"[{n}/{len(items)}] {item['id']} 실패: {e}", flush=True)
                continue
            record = to_record(item, text, args.model, sec)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            counts["saved"] += 1
            counts["json_error"] += record["errors"] == ["json"]
            counts["format_error"] += bool(record["errors"]) and record["errors"] != ["json"]
            total_sec += sec
            if n % 10 == 0 or n == len(items):
                print(f"[{n}/{len(items)}] 평균 {total_sec / counts['saved']:.1f}초/문구", flush=True)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
