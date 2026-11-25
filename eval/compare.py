"""여러 평가 결과(summary.json)를 한 표로 비교한다.

사용: .venv/Scripts/python.exe -m eval.compare            (results 폴더 전체)
      .venv/Scripts/python.exe -m eval.compare 폴더1 폴더2  (원하는 결과만)
"""
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

KEYS = [
    "KTAS 정확 일치율",
    "KTAS ±1 이내율",
    "Under-triage율 (과소분류)",
    "Over-triage율 (과대분류)",
    "위급 환자 과소분류율 (KTAS 1~2, 목표 0%)",
    "진료과 1순위 적중률",
    "질환 코드 적중률 (필수 케이스)",
    "되묻기 케이스 정답률",
    "불필요한 되묻기율",
]


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in RESULTS.iterdir() if p.is_dir())
    runs = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in dirs if (d / "summary.json").exists()]
    if not runs:
        print("비교할 결과가 없습니다. 먼저 eval.run_eval 을 실행하세요.")
        return
    lines = ["| 지표 | " + " | ".join(r["label"] for r in runs) + " |",
             "|---|" + "---|" * len(runs)]
    for k in KEYS:
        lines.append(f"| {k} | " + " | ".join(f"{r['metrics'][k]['rate'] * 100:.1f}%" for r in runs) + " |")
    lines.append("| 평균 응답 시간 | " + " | ".join(f"{r['latency']['mean_sec']}초" for r in runs) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
