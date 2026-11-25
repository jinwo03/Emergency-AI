"""멘토링 피드백 대응: LLM 대신 분류 모델을 학습시켜 비교한다.

개발용 시나리오(scenarios.csv)로 학습하고, 검증용(scenarios_holdout.csv)으로 예측해
run_eval --replay 가 읽을 수 있는 예측 파일을 만든다. 채점은 run_eval이 똑같은 기준으로 한다.

모델
- tfidf : 글자 n-gram TF-IDF + 로지스틱 회귀 (전통적인 NLP 기준선, GPU 불필요, 몇 초)
- klue  : KLUE-RoBERTa-base 파인튜닝 (한국어 사전학습 모델, GPU 권장, 4060 Ti 기준 수 분)

사용 예 (Windows PowerShell, 저장소 루트에서)
    .venv/Scripts/python.exe -m eval.finetune --model tfidf
    .venv/Scripts/python.exe -m eval.run_eval --set holdout --replay eval/finetune_out/tfidf_holdout.csv --name tfidf+규칙_holdout
    .venv/Scripts/python.exe -m eval.run_eval --set holdout --replay eval/finetune_out/tfidf_holdout.csv --no-safety --name tfidf단독_holdout

    uv pip install --python .venv/Scripts/python.exe torch transformers --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
    .venv/Scripts/python.exe -m eval.finetune --model klue
    (위와 같은 run_eval 두 줄에서 tfidf 를 klue 로 바꿔 실행)

예측 대상
- KTAS (1~5 중 하나): 다중 클래스 분류
- 진료과, 질환 코드: 다중 레이블 분류 (정답 목록에 있는 것은 모두 정답으로 학습)
되묻기 케이스(정답 KTAS 없음)는 학습에서 제외한다. 되묻기 판단은 v2 규칙이 그대로 맡는다.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from algorithm.egen_codes import DEPARTMENTS, EGEN_CONDITIONS
from eval.run_eval import SCENARIOS, SETS, load_scenarios

OUT_DIR = Path(__file__).resolve().parent / "finetune_out"
KTAS_LEVELS = [1, 2, 3, 4, 5]
CODES = list(EGEN_CONDITIONS)
COND_THRESHOLD = 0.5
MAX_DEPTS = 3


def model_input(sc: dict) -> str:
    """나이를 문장 앞에 붙여 모델 입력으로 쓴다 (나이가 KTAS·진료과에 영향을 주기 때문)"""
    age = "나이 모름" if sc["age"] is None else f"{sc['age']}세"
    return f"{age} {sc['text']}"


def multi_hot(labels: list[str], vocab: list[str]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    for x in labels:
        if x in vocab:
            v[vocab.index(x)] = 1.0
    return v


def decode(ktas_probs, dept_probs, cond_probs) -> dict:
    """확률 → 예측 (KTAS 최댓값, 진료과 상위 3개, 질환 코드 임계값 이상)"""
    ktas = KTAS_LEVELS[int(np.argmax(ktas_probs))]
    depts = [DEPARTMENTS[i] for i in np.argsort(-dept_probs)[:MAX_DEPTS]]
    conds = [CODES[i] for i in np.argsort(-cond_probs) if cond_probs[i] >= COND_THRESHOLD][:3]
    return {"ktas": ktas, "departments": depts, "conditions": conds}


# ── 전통적인 기준선: TF-IDF + 로지스틱 회귀 ───────────────────────────
def run_tfidf(train: list[dict], test: list[dict]) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=1, sublinear_tf=True)
    x_train = vec.fit_transform([model_input(s) for s in train])
    x_test = vec.transform([model_input(s) for s in test])

    ktas_clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
    ktas_clf.fit(x_train, [s["ktas"] for s in train])
    ktas_p = np.zeros((len(test), len(KTAS_LEVELS)))
    for j, c in enumerate(ktas_clf.classes_):
        ktas_p[:, KTAS_LEVELS.index(int(c))] = ktas_clf.predict_proba(x_test)[:, j]

    def multilabel(vocab, key):
        y = np.stack([multi_hot(s[key], vocab) for s in train])
        probs = np.zeros((len(test), len(vocab)))
        cols = [i for i in range(len(vocab)) if 0 < y[:, i].sum() < len(train)]   # 한 번도 안 나온 레이블은 학습 불가
        if cols:
            clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced"))
            clf.fit(x_train, y[:, cols])
            probs[:, cols] = clf.predict_proba(x_test)
        return probs

    dept_p = multilabel(DEPARTMENTS, "departments")
    cond_p = multilabel(CODES, "conditions")
    return [decode(ktas_p[i], dept_p[i], cond_p[i]) for i in range(len(test))]


# ── KLUE-RoBERTa 파인튜닝 ────────────────────────────────────────────
def run_klue(train: list[dict], test: list[dict], epochs: int, seed: int) -> list[dict]:
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "klue/roberta-base"
    tok = AutoTokenizer.from_pretrained(name)

    class Heads(nn.Module):
        """하나의 인코더 위에 KTAS·진료과·질환 코드 출력층 3개"""
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(name)
            h = self.enc.config.hidden_size
            self.drop = nn.Dropout(0.1)
            self.ktas = nn.Linear(h, len(KTAS_LEVELS))
            self.dept = nn.Linear(h, len(DEPARTMENTS))
            self.cond = nn.Linear(h, len(CODES))

        def forward(self, **batch):
            cls = self.drop(self.enc(**batch).last_hidden_state[:, 0])
            return self.ktas(cls), self.dept(cls), self.cond(cls)

    model = Heads().to(device)
    enc = lambda rows: tok([model_input(s) for s in rows], padding=True, truncation=True,
                           max_length=64, return_tensors="pt").to(device)

    y_ktas = torch.tensor([KTAS_LEVELS.index(s["ktas"]) for s in train], device=device)
    y_dept = torch.tensor(np.stack([multi_hot(s["departments"], DEPARTMENTS) for s in train]), device=device)
    y_cond = torch.tensor(np.stack([multi_hot(s["conditions"], CODES) for s in train]), device=device)

    # KTAS 클래스 불균형 보정 (2가 많고 1·5가 적음)
    counts = torch.bincount(y_ktas, minlength=len(KTAS_LEVELS)).float().clamp(min=1)
    ce = nn.CrossEntropyLoss(weight=(counts.sum() / counts) / len(KTAS_LEVELS))
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    batch_size = 16
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(train))
        total = 0.0
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            out_k, out_d, out_c = model(**enc([train[i] for i in idx]))
            loss = ce(out_k, y_ktas[idx]) + bce(out_d, y_dept[idx]) + bce(out_c, y_cond[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {ep + 1}/{epochs} loss {total / len(train):.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(test), 32):
            rows = test[k:k + 32]
            out_k, out_d, out_c = model(**enc(rows))
            pk = out_k.softmax(-1).cpu().numpy()
            pd = out_d.sigmoid().cpu().numpy()
            pc = out_c.sigmoid().cpu().numpy()
            preds += [decode(pk[i], pd[i], pc[i]) for i in range(len(rows))]
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="분류 모델 학습 후 검증 세트 예측 (run_eval --replay 용)")
    ap.add_argument("--model", choices=["tfidf", "klue"], default="tfidf")
    ap.add_argument("--test", choices=list(SETS), default="holdout")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = [s for s in load_scenarios(SCENARIOS) if s["ktas"] is not None]
    test = load_scenarios(SETS[args.test])
    print(f"학습 {len(train)}건, 예측 {len(test)}건, 모델 {args.model}")

    preds = run_tfidf(train, test) if args.model == "tfidf" else run_klue(train, test, args.epochs, args.seed)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{args.model}_{args.test}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["text", "pred_ktas", "pred_departments", "pred_conditions"])
        for sc, p in zip(test, preds):
            w.writerow([sc["text"], p["ktas"], "|".join(p["departments"]), "|".join(p["conditions"])])
    print(f"저장: {out}")
    print(f"채점: python -m eval.run_eval --set {args.test} --replay {out.as_posix()} [--no-safety]")


if __name__ == "__main__":
    main()
