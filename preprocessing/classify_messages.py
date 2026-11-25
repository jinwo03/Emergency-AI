"""응급 메시지 문구를 분류 모델로 해석한다 (방식 A: 기존 NLP 분류기). LLM 해석기와 같은 형식으로 저장해 비교한다.

실행:
    python -m preprocessing.classify_messages --backend tfidf      # TF-IDF + 로지스틱 회귀 (CPU, 수 초)
    python -m preprocessing.classify_messages --backend roberta    # KLUE-RoBERTa 파인튜닝 (GPU 권장)
    python -m preprocessing.classify_messages --backend rule       # 문구 규칙 (기준점)
    python -m preprocessing.classify_messages --backend oracle     # 정답을 한 줄로 줄였을 때의 상한
    python -m preprocessing.score_parses --provisional --pred data/labels/cls_parse_<backend>.jsonl

문제를 이렇게 단순화한다 (여러 줄 구조는 분류기로 다루기 어렵다)
- 문구마다 한 줄만 만든다. 상태는 줄들 중 가장 강한 제한(불가 > 기존 환자만 > 어려움 > 가능), 모든 줄이
  판단 불가일 때만 판단 불가. 진료과·질환·조건·주의는 줄들의 합집합. 시간은 학습하지 않고 '항상'.
- 과제: 상태(5종 중 하나), 사유(8종 + 없음 중 하나), 진료과·질환·조건·주의(여러 개 선택).
- 학습: 정답 시트의 학습용 답. 학습용 끝 15%(ID 순)는 검증용(RoBERTa 조기 종료). 시험: 시험용 문구.
- oracle 은 시험용 정답을 같은 방식으로 한 줄로 줄인 것이다. 이 방식이 낼 수 있는 점수의 상한이다.

결과 파일은 시험용 해석을 담으므로 사람이 시험용 시트를 채우기 전에는 열지 않는다.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.vocab import ALL_DEPARTMENTS, DEPARTMENTS, DISEASE_CODES, PATIENT_CONDITIONS
from egen import store
from preprocessing import label_set, message_status, message_time

STATUS, DEPT, CODES, CONDS, CAUTION = (label_set.FIELDS.index(f) for f in ("상태", "진료과", "질환", "조건", "주의"))
SEVERITY = {"불가": 4, "기존 환자만": 3, "어려움": 2, "가능": 1}
NO_REASON = "없음"
CAUTION_PARTS = ["문의", "지연"]
SINGLE = {"status": label_set.STATUSES, "reason": [NO_REASON, *label_set.REASONS]}
MULTI = {"depts": [ALL_DEPARTMENTS, *DEPARTMENTS], "codes": list(DISEASE_CODES),
         "conds": list(PATIENT_CONDITIONS), "caution": CAUTION_PARTS}
VALID_RATIO = 0.15
ROBERTA = "klue/roberta-base"
ENCODER_LR, HEAD_LR = 3e-5, 1e-3   # 출력층은 처음부터 배우므로 크게
POS_WEIGHT_MAX = 10.0
MIN_CONFIDENCE = 0.6                   # 상태 확률이 이보다 낮으면 판단 불가
MODEL_DIR = store.DEFAULT_DB.parent / "models" / "message_classifier"
APPLY_FILE = "message_cls.jsonl"       # 정답 시트에 없는 문구의 분류 결과 (message_answers.CLASSIFIER_FILE 과 같아야 함)


# ---------- 정답 ↔ 과제 값 ----------

def encode(answer: dict) -> dict:
    """정답(여러 줄) → 과제 값 {status, reason, depts, codes, conds, caution}."""
    lines = answer["lines"]
    known = [l[STATUS] for l in lines if l[STATUS] in SEVERITY]
    status = max(known, key=SEVERITY.get) if known else message_status.UNKNOWN
    union = lambda i: sorted({v for l in lines for v in label_set.split_values(l[i])})
    caution = {p for l in lines for p in CAUTION_PARTS if p in (l[CAUTION] or "")}
    depts = union(DEPT)
    if ALL_DEPARTMENTS in depts and len(depts) > 1:
        depts = [ALL_DEPARTMENTS]
    return {"status": status, "reason": answer.get("reason") or NO_REASON, "depts": depts,
            "codes": union(CODES), "conds": union(CONDS), "caution": sorted(caution)}


def decode(v: dict, item: dict, use_time: bool = True) -> dict:
    """과제 값 → 정답 형식. 진료과가 비면 메시지 진료과 항목(목록에 없으면 전체).

    시간은 원문 규칙(message_time)으로 붙인다: 제한 꼴이면 한 줄, 허용 꼴이면 (시간 가능 / 그 외 상태) 두 줄.
    """
    depts = [d for d in v["depts"] if d != ALL_DEPARTMENTS] or ([ALL_DEPARTMENTS] if ALL_DEPARTMENTS in v["depts"] else [])
    if not depts:
        depts = [item["dept"]] if item.get("dept") in DEPARTMENTS else [ALL_DEPARTMENTS]
    caution = "·".join(p for p in CAUTION_PARTS if p in v["caution"])
    sep = label_set.SEP
    head = [item["msg"][:20].strip() or "메시지", sep.join(depts), sep.join(v["codes"]), sep.join(v["conds"])]
    times = message_time.extract(item["msg"], v["status"]) if use_time else None
    if times:   # 허용 꼴의 '가능' 줄에는 주의를 붙이지 않는다
        lines = [[*head, t, s, "" if s == "가능" and len(times) > 1 else caution] for t, s in times]
    else:
        lines = [[*head, "항상", v["status"], caution]]
    return {"reason": None if v["reason"] == NO_REASON else v["reason"], "lines": lines, "memo": ""}


def to_record(item: dict, answer: dict, backend: str) -> dict:
    return {"msg_norm": item["msg_norm"], "msg": item["msg"], "model": f"cls_{backend}", "prompt_version": backend,
            "parsed_at": store.iso(store.now_kst()), **answer, "confidence": 1.0,
            "errors": label_set.validate_answer(answer)}


# ---------- 데이터 ----------

def text_of(item: dict) -> str:
    return f"{item.get('dept') or ''} | {item['msg']}"


def load_data(labels_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """(학습, 검증, 시험). 학습·검증에는 target(과제 값)이 붙는다. 시험은 정답 없이 문구만."""
    answers, _ = label_set.load_label_answers(labels_dir)
    phrases = pd.read_parquet(labels_dir / label_set.PHRASE_FILE, columns=["id", "msg_norm", "msg", "dept", "split"])
    rows = [{"id": p.id, "msg_norm": p.msg_norm, "msg": p.msg, "dept": p.dept, "split": p.split}
            for p in phrases.sort_values("id").itertuples()]
    train = [r | {"target": encode(answers[r["msg_norm"]])} for r in rows
             if r["split"] == "train" and r["msg_norm"] in answers]
    cut = int(len(train) * (1 - VALID_RATIO))
    test = [r for r in rows if r["split"] == "test"]
    return train[:cut], train[cut:], test


def multi_hot(values: list[str], classes: list[str]) -> np.ndarray:
    return np.array([c in values for c in classes], dtype=np.float32)


# ---------- 모델 ----------

def predict_rule(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        s = message_status.general_status(it["msg"])
        out.append({"status": message_status.UNKNOWN if s == message_status.UNPARSED else s, "reason": NO_REASON,
                    "depts": [], "codes": [], "conds": [], "caution": ["문의"] if "문의" in it["msg"] else []})
    return out


def fit_predict_tfidf(train: list[dict], items: list[dict]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=1, sublinear_tf=True)
    X = vec.fit_transform([text_of(r) for r in train])
    Xt = vec.transform([text_of(r) for r in items])
    preds = [dict() for _ in items]
    for task in SINGLE:
        y = [r["target"][task] for r in train]
        if len(set(y)) == 1:
            for p in preds:
                p[task] = y[0]
            continue
        clf = LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(X, y)
        for p, v in zip(preds, clf.predict(Xt)):
            p[task] = v
    for task, classes in MULTI.items():
        for p in preds:
            p[task] = []
        for c in classes:
            y = [c in r["target"][task] for r in train]
            if sum(y) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(X, y)
            for p, hit in zip(preds, clf.predict(Xt)):
                if hit:
                    p[task].append(c)
    return preds


def _net(model_name_or_dir: str):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(model_name_or_dir)
            h = self.enc.config.hidden_size
            self.drop = nn.Dropout(0.1)
            self.heads = nn.ModuleDict({t: nn.Linear(h, len(c)) for t, c in {**SINGLE, **MULTI}.items()})

        def forward(self, **x):
            z = self.drop(self.enc(**x).last_hidden_state[:, 0])
            return {t: head(z) for t, head in self.heads.items()}

    return Net()


def _batches(tok, rows, size, shuffle, device):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), size):
        part = [rows[j] for j in idx[i:i + size]]
        x = tok([text_of(r) for r in part], padding=True, truncation=True, max_length=128, return_tensors="pt")
        yield part, {k: v.to(device) for k, v in x.items()}


def train_roberta(train: list[dict], valid: list[dict], epochs: int = 25, seed: int = 0, model_name: str = ROBERTA):
    """(모델, 토크나이저, 정보). 검증 손실이 가장 낮은 에폭의 가중치를 쓴다."""
    import torch
    from torch import nn
    from transformers import AutoTokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)

    def targets(part):
        y = {t: torch.tensor([c.index(r["target"][t]) for r in part], device=device) for t, c in SINGLE.items()}
        y |= {t: torch.tensor(np.stack([multi_hot(r["target"][t], c) for r in part]), device=device)
              for t, c in MULTI.items()}
        return y

    # 여러 개 선택 과제는 대부분 0 이라 그냥 학습하면 아무것도 고르지 않는다 → 드문 값에 가중치 (최대 POS_WEIGHT_MAX)
    ce = nn.CrossEntropyLoss()
    bce = {}
    for t, c in MULTI.items():
        pos = np.stack([multi_hot(r["target"][t], c) for r in train]).sum(0)
        w = np.clip((len(train) - pos) / np.maximum(pos, 1), 1, POS_WEIGHT_MAX)
        bce[t] = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, dtype=torch.float32, device=device))

    def loss_of(out, y):
        return sum(ce(out[t], y[t]) for t in SINGLE) + sum(bce[t](out[t], y[t]) for t in MULTI)

    net = _net(model_name).to(device)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": ENCODER_LR},
                             {"params": net.heads.parameters(), "lr": HEAD_LR}], weight_decay=0.01)
    steps = epochs * ((len(train) + 15) // 16)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.1 * steps)) * max(0.0, 1 - s / steps))
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        net.train()
        for part, x in _batches(tok, train, 16, True, device):
            loss = loss_of(net(**x), targets(part))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            vl = sum(loss_of(net(**x), targets(part)).item() * len(part)
                     for part, x in _batches(tok, valid, 64, False, device)) / len(valid)
        history.append(round(vl, 4))
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, tok, {"device": device, "valid_loss": history, "best_epoch": int(np.argmin(history)) + 1}


def predict_roberta(net, tok, items: list[dict]) -> list[dict]:
    """과제 값 + confidence(상태 확률 중 가장 큰 값)."""
    import torch

    device = next(net.parameters()).device
    preds = []
    with torch.no_grad():
        for part, x in _batches(tok, items, 64, False, device):
            out = net(**x)
            status_prob = torch.softmax(out["status"], -1)
            for i in range(len(part)):
                p = {t: c[int(out[t][i].argmax())] for t, c in SINGLE.items()}
                p |= {t: [cl for cl, z in zip(c, out[t][i].tolist()) if z > 0] for t, c in MULTI.items()}
                p["confidence"] = round(float(status_prob[i].max()), 4)
                preds.append(p)
    return preds


def save_roberta(net, tok, out_dir: Path, meta: dict) -> None:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    net.enc.save_pretrained(out_dir / "encoder")
    tok.save_pretrained(out_dir / "encoder")
    torch.save(net.heads.state_dict(), out_dir / "heads.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta | {"single": SINGLE, "multi": MULTI}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")


def load_roberta(model_dir: Path):
    """(모델, 토크나이저, 정보). 저장할 때와 과제 목록이 다르면 오류."""
    import torch
    from transformers import AutoTokenizer

    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["single"] != SINGLE or meta["multi"] != MULTI:
        raise ValueError("저장된 모델의 분류 목록이 지금 코드와 다릅니다. 다시 학습하세요 (--backend roberta --save)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _net(str(model_dir / "encoder"))
    net.heads.load_state_dict(torch.load(model_dir / "heads.pt", map_location="cpu"))
    net.to(device).eval()
    return net, AutoTokenizer.from_pretrained(model_dir / "encoder"), meta


def apply_confidence(p: dict, threshold: float = MIN_CONFIDENCE) -> dict:
    """상태 확률이 기준보다 낮으면 '판단 불가' (병원을 빼지 않고 원문을 보여 준다)."""
    if p.get("confidence", 1.0) < threshold:
        return p | {"status": message_status.UNKNOWN}
    return p


# ---------- 과제별 지표 ----------

def task_metrics(golds: list[dict], preds: list[dict]) -> dict:
    """상태·사유 정확도, 상태 macro-F1, 여러 개 선택 과제의 micro-F1."""
    from sklearn.metrics import f1_score

    out = {}
    for t in SINGLE:
        g, p = [x[t] for x in golds], [x[t] for x in preds]
        out[f"{t}_acc"] = round(float(np.mean([a == b for a, b in zip(g, p)])), 4)
    out["status_macro_f1"] = round(float(f1_score([x["status"] for x in golds], [x["status"] for x in preds],
                                                   average="macro", zero_division=0)), 4)
    out["status_blocked_recall"] = _recall(golds, preds, {"불가", "기존 환자만"})
    for t, c in MULTI.items():
        tp = sum(len(set(g[t]) & set(p[t])) for g, p in zip(golds, preds))
        n_pred, n_gold = sum(len(p[t]) for p in preds), sum(len(g[t]) for g in golds)
        out[f"{t}_micro_f1"] = round(2 * tp / (n_pred + n_gold), 4) if n_pred + n_gold else None
    return out


def _recall(golds, preds, group) -> float | None:
    """정답이 못 받음(불가·기존 환자만)인 문구 중 예측도 못 받음인 비율. 안전에 가장 중요한 값."""
    hits = [p["status"] in group for g, p in zip(golds, preds) if g["status"] in group]
    return round(float(np.mean(hits)), 4) if hits else None


# ---------- 실행 ----------

def unlabeled_items(labels_dir: Path, db_path: Path | str | None = None) -> list[dict]:
    """수집 DB 의 응급 메시지 고유 문구 중 정답 시트에 답이 없는 것 (개인정보는 이미 가려진 값)."""
    from preprocessing import clean

    with clean.open_readonly(db_path) as conn:
        messages, _ = clean.clean_messages(conn)
    labels, _ = label_set.load_label_answers(labels_dir)
    general = messages[(messages["msg_type"] == "응급") & ~messages["msg_norm"].isin(labels)].drop_duplicates("msg_norm")
    return [{"msg_norm": m.msg_norm, "msg": m.msg, "dept": m.dept} for m in general.itertuples()]


def write_records(path: Path, items: list[dict], preds: list[dict], backend: str, version: str | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item, p in zip(items, preds):
            record = to_record(item, decode(p, item), backend) | {"confidence": p.get("confidence", 1.0)}
            if version:
                record["prompt_version"] = version
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_saved(labels_dir: Path, model_dir: Path) -> dict:
    net, tok, meta = load_roberta(model_dir)
    items = unlabeled_items(labels_dir)
    preds = [apply_confidence(p) for p in predict_roberta(net, tok, items)]
    path = labels_dir / APPLY_FILE
    write_records(path, items, preds, "roberta", meta["trained_at"])
    return {"model": str(model_dir), "trained_at": meta["trained_at"], "phrases": len(items),
            "low_confidence": sum(p["status"] == message_status.UNKNOWN for p in preds), "saved": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["rule", "tfidf", "roberta", "oracle"], default="tfidf")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true", help=f"roberta 학습 모델을 저장 ({MODEL_DIR})")
    parser.add_argument("--apply", action="store_true", help=f"저장된 모델로 정답 시트에 없는 문구 분류 → {APPLY_FILE}")
    parser.add_argument("--no-time", action="store_true", help="시간 규칙 없이 (비교용)")
    parser.add_argument("--labels", type=Path, default=store.DEFAULT_DB.parent / "labels")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.apply:
        print(json.dumps(apply_saved(args.labels, args.model_dir), ensure_ascii=False, indent=2))
        return

    train, valid, test = load_data(args.labels)
    gold_answers = label_set.load_provisional_answers(args.labels) | label_set.load_label_answers(args.labels)[0]
    start = time.perf_counter()
    info, net = {}, None
    if args.backend == "rule":
        preds = predict_rule(test)
    elif args.backend == "tfidf":
        preds = fit_predict_tfidf(train + valid, test)
    elif args.backend == "roberta":
        net, tok, info = train_roberta(train, valid, args.epochs, args.seed)
        preds = predict_roberta(net, tok, test)
    else:
        preds = [encode(gold_answers[t["msg_norm"]]) if t["msg_norm"] in gold_answers else predict_rule([t])[0]
                 for t in test]
    seconds = time.perf_counter() - start

    raw = preds
    preds = [apply_confidence(p) for p in preds]
    path = args.labels / f"cls_parse_{args.backend}{'_notime' if args.no_time else ''}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item, p in zip(test, preds):
            record = to_record(item, decode(p, item, not args.no_time), args.backend)
            f.write(json.dumps(record | {"confidence": p.get("confidence", 1.0)}, ensure_ascii=False) + "\n")

    pairs = [(encode(gold_answers[t["msg_norm"]]), r, p) for t, r, p in zip(test, raw, preds) if t["msg_norm"] in gold_answers]
    report = {"backend": args.backend, "train": len(train), "valid": len(valid), "test": len(test),
              "test_with_gold": len(pairs), "seconds": round(seconds, 1), **info,
              "low_confidence": sum(p["status"] == message_status.UNKNOWN and r["status"] != message_status.UNKNOWN
                                    for _, r, p in pairs),
              "test_tasks_raw": task_metrics([g for g, _, _ in pairs], [r for _, r, _ in pairs]) if pairs else None,
              "test_tasks": task_metrics([g for g, _, _ in pairs], [p for _, _, p in pairs]) if pairs else None,
              "saved": str(path)}
    if args.save and net is not None:
        save_roberta(net, tok, args.model_dir, {"trained_at": store.iso(store.now_kst()), "base_model": ROBERTA,
                                                "seed": args.seed, "train": len(train), "valid": len(valid),
                                                "min_confidence": MIN_CONFIDENCE, **info,
                                                "test_tasks_provisional": report["test_tasks"]})
        report["model_saved"] = str(args.model_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
