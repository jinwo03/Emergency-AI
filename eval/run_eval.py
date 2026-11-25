"""증상 분석(triage) 정확도 평가.

eval/scenarios.csv의 정답과 analyze_symptoms() 결과를 비교해 지표를 계산한다.

사용 예 (Windows PowerShell, 저장소 루트에서)
    .venv/Scripts/python.exe -m eval.run_eval --backend rule
    .venv/Scripts/python.exe -m eval.run_eval --backend llm
    .venv/Scripts/python.exe -m eval.run_eval --backend llm --no-safety
    .venv/Scripts/python.exe -m eval.run_eval --backend llm --model exaone3.5:7.8b
    .venv/Scripts/python.exe -m eval.run_eval --replay eval/v1_run/cases_qwen_v1.csv
      (이전 LLM 출력을 다시 써서 후처리 규칙만 바꾼 효과를 빠르게 확인, Ollama 불필요)

결과는 eval/results/<시각>_<이름>/ 아래 cases.csv, summary.md, summary.json 으로 저장된다.
"""
import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

from algorithm import triage
from contracts.schemas import TriageStatus

ROOT = Path(__file__).resolve().parent
SCENARIOS = ROOT / "scenarios.csv"            # 개발용: 오류 분석·규칙 튜닝에 사용
HOLDOUT = ROOT / "scenarios_holdout.csv"       # 검증용: 튜닝에 쓰지 않고 최종 성능 보고에만 사용
STRESS = ROOT / "scenarios_stress.csv"         # 구어체 스트레스: 횡설수설·음성인식 오류·함정 표현 (튜닝 금지)
SETS = {"dev": SCENARIOS, "holdout": HOLDOUT, "stress": STRESS}


def _split(value: str) -> list[str]:
    return [v for v in (value or "").split("|") if v]


def make_replay(path: Path):
    """이전 평가의 LLM 출력(cases.csv)을 그대로 돌려주는 가짜 LLM을 만든다"""
    recorded = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            recorded[r["text"]] = {
                "ktas": int(r["pred_ktas"]) if r["pred_ktas"] else 3,
                "departments": _split(r["pred_departments"]),
                "conditions": _split(r["pred_conditions"]),
            }

    def fake(text, age, history):
        return recorded.get(text, {"ktas": 3, "departments": [], "conditions": []})
    return fake


def load_scenarios(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"],
                "category": r["category"],
                "text": r["text"],
                "age": int(r["age"]) if r["age"] else None,
                "ktas": int(r["ktas"]) if r["ktas"] else None,   # 비어 있으면 되묻기 기대
                "departments": _split(r["departments"]),
                "conditions": _split(r["conditions"]),
                "cond_required": r["cond_required"].strip().upper() == "Y",
                "note": r.get("note", ""),
            })
    return rows


def evaluate_case(sc: dict) -> dict:
    """한 시나리오를 실행하고 정답과 비교한 판정 결과를 돌려준다"""
    start = time.perf_counter()
    res = triage.analyze_symptoms(sc["text"], sc["age"], [])
    sec = time.perf_counter() - start

    asked = res.status == TriageStatus.need_more_info
    expect_ask = sc["ktas"] is None
    out = {
        "id": sc["id"], "category": sc["category"], "text": sc["text"], "age": sc["age"],
        "gold_ktas": sc["ktas"], "pred_ktas": res.ktas,
        "gold_departments": "|".join(sc["departments"]), "pred_departments": "|".join(res.departments),
        "gold_conditions": "|".join(sc["conditions"]), "pred_conditions": "|".join(res.conditions),
        "follow_up_question": res.follow_up_question or "",
        "seconds": round(sec, 2),
        "source": getattr(triage, "LAST_SOURCE", "?"),
        "status_ok": asked == expect_ask,
    }
    if expect_ask:
        return out

    gold, pred = sc["ktas"], res.ktas
    out["unneeded_ask"] = asked
    out["ktas_exact"] = (not asked) and pred == gold
    out["ktas_within1"] = (not asked) and abs(pred - gold) <= 1
    out["under"] = (not asked) and pred > gold          # 실제보다 덜 위급하게 판단
    out["over"] = (not asked) and pred < gold           # 실제보다 더 위급하게 판단
    # 안전 지표: 위급(KTAS 1~2) 환자를 3 이상으로 보거나 되묻느라 지연
    out["critical_under"] = gold <= 2 and (asked or pred >= 3)

    if not asked:
        out["dept_top1"] = bool(res.departments) and res.departments[0] in sc["departments"]
        out["dept_any"] = any(d in sc["departments"] for d in res.departments)
        allowed = set(sc["conditions"])
        out["cond_hit"] = any(c in allowed for c in res.conditions) if sc["cond_required"] else None
        out["cond_outside"] = any(c not in allowed for c in res.conditions)   # 허용 밖 질환 코드 출력
    return out


def _rate(rows: list[dict], key: str) -> tuple[float, int, int]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    hit = sum(1 for v in vals if v)
    return (hit / len(vals) if vals else 0.0), hit, len(vals)


def summarize(results: list[dict], label: str) -> dict:
    graded = [r for r in results if r["gold_ktas"] is not None]
    ask_cases = [r for r in results if r["gold_ktas"] is None]
    critical = [r for r in graded if r["gold_ktas"] <= 2]
    secs = sorted(r["seconds"] for r in results)

    metrics = {
        "상태 판단 정확도": _rate(results, "status_ok"),
        "되묻기 케이스 정답률": _rate(ask_cases, "status_ok"),
        "불필요한 되묻기율": _rate(graded, "unneeded_ask"),
        "KTAS 정확 일치율": _rate(graded, "ktas_exact"),
        "KTAS ±1 이내율": _rate(graded, "ktas_within1"),
        "Under-triage율 (과소분류)": _rate(graded, "under"),
        "Over-triage율 (과대분류)": _rate(graded, "over"),
        "위급 환자 과소분류율 (KTAS 1~2, 목표 0%)": _rate(critical, "critical_under"),
        "진료과 1순위 적중률": _rate(graded, "dept_top1"),
        "진료과 포함 적중률": _rate(graded, "dept_any"),
        "질환 코드 적중률 (필수 케이스)": _rate(graded, "cond_hit"),
        "허용 밖 질환 코드 출력률": _rate(graded, "cond_outside"),
    }

    # KTAS 혼동행렬: 행=정답, 열=예측(1~5, ?=되묻기)
    cols = ["1", "2", "3", "4", "5", "?"]
    matrix = {g: {c: 0 for c in cols} for g in ["1", "2", "3", "4", "5"]}
    for r in graded:
        p = "?" if r["unneeded_ask"] else str(r["pred_ktas"])
        matrix[str(r["gold_ktas"])][p] += 1

    return {
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "n": len(results),
        "metrics": {k: {"rate": round(v[0], 4), "hit": v[1], "total": v[2]} for k, v in metrics.items()},
        "llm_fallback": sum(1 for r in results if r.get("source") == "rule"),
        "latency": {
            "mean_sec": round(statistics.mean(secs), 2) if secs else 0,
            "p95_sec": round(secs[int(len(secs) * 0.95) - 1], 2) if secs else 0,
        },
        "ktas_confusion": matrix,
        "critical_misses": [
            {"id": r["id"], "text": r["text"], "gold": r["gold_ktas"],
             "pred": "되묻기" if r["unneeded_ask"] else r["pred_ktas"]}
            for r in critical if r["critical_under"]
        ],
    }


def to_markdown(s: dict) -> str:
    lines = [f"# Triage 평가 결과: {s['label']}", "",
             f"- 시각: {s['time']}  ",
             f"- 시나리오 수: {s['n']}  ",
             f"- 응답 시간: 평균 {s['latency']['mean_sec']}초, p95 {s['latency']['p95_sec']}초  ",
             f"- LLM 실패로 규칙 대체: {s.get('llm_fallback', 0)}건", "",
             "| 지표 | 값 | 맞음/전체 |", "|---|---|---|"]
    for k, v in s["metrics"].items():
        lines.append(f"| {k} | {v['rate'] * 100:.1f}% | {v['hit']}/{v['total']} |")
    lines += ["", "## KTAS 혼동행렬 (행=정답, 열=예측, ?=되묻기)", "",
              "| 정답＼예측 | 1 | 2 | 3 | 4 | 5 | ? |", "|---|---|---|---|---|---|---|"]
    for g, row in s["ktas_confusion"].items():
        lines.append(f"| {g} | " + " | ".join(str(row[c]) for c in ["1", "2", "3", "4", "5", "?"]) + " |")
    lines += ["", f"## 위급 환자 과소분류 목록 ({len(s['critical_misses'])}건)", ""]
    for m in s["critical_misses"]:
        lines.append(f"- {m['id']} 정답 {m['gold']} → 예측 {m['pred']}: {m['text']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="triage 정확도 평가")
    ap.add_argument("--backend", choices=["rule", "llm"], default="llm")
    ap.add_argument("--model", help="Ollama 모델 이름 (기본: TRIAGE_MODEL 또는 qwen2.5:7b)")
    ap.add_argument("--no-safety", action="store_true", help="위험 신호 KTAS 상한 끄기 (비교 실험용)")
    ap.add_argument("--limit", type=int, help="앞에서 N건만 실행 (빠른 확인용)")
    ap.add_argument("--name", help="결과 폴더 이름")
    ap.add_argument("--set", choices=list(SETS), default="dev",
                    help="dev=개발용 201건(규칙 튜닝에 사용), holdout=검증용(튜닝에 안 쓴 문제), stress=구어체 스트레스")
    ap.add_argument("--replay", help="이전 cases.csv의 LLM 출력을 재사용 (Ollama 없이 후처리만 비교)")
    args = ap.parse_args()

    os.environ["TRIAGE_BACKEND"] = args.backend
    if args.model:
        os.environ["TRIAGE_MODEL"] = args.model
    if args.no_safety:
        # 규칙 보정을 전부 끔: 규칙 코드·상한 없음, 질환 코드 상한 없음
        triage.rules.scan = lambda text, age: ([], None)
        triage.rules.CODE_CEILING = {k: 5 for k in triage.rules.CODE_CEILING}
    if args.replay:
        os.environ["TRIAGE_BACKEND"] = "llm"
        triage._call_llm = make_replay(Path(args.replay))

    model = os.environ.get("TRIAGE_MODEL", "qwen2.5:7b") if args.backend == "llm" else "rule"
    if args.replay:
        model = "replay-" + Path(args.replay).stem
    label = args.name or (model.replace(":", "-") + ("_nosafety" if args.no_safety else "") + f"_{args.set}")

    if os.environ["TRIAGE_BACKEND"] == "llm" and not args.replay:
        print("모델 준비 중 (첫 호출 워밍업)...")
        triage.analyze_symptoms("가슴이 아파요", 50, [])

    scenarios = load_scenarios(SETS[args.set])
    if args.limit:
        scenarios = scenarios[: args.limit]

    results = []
    for i, sc in enumerate(scenarios, 1):
        r = evaluate_case(sc)
        results.append(r)
        mark = "" if r["status_ok"] and r.get("critical_under") is not True else "  ← 확인"
        print(f"[{i:3d}/{len(scenarios)}] {r['id']} 정답 {r['gold_ktas'] or '되묻기'} → "
              f"예측 {r['pred_ktas'] or '되묻기'} ({r['seconds']}초){mark}")

    summary = summarize(results, label)
    out_dir = ROOT / "results" / f"{datetime.now():%Y%m%d_%H%M}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cases.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = list(dict.fromkeys(k for r in results for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(summary)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
