"""전처리 1~3단계 실행. 원본 DB 는 읽기만 하고 결과는 출력 폴더에만 쓴다.

실행:
    python -m preprocessing.run                # 결과: data/processed/
    python -m preprocessing.run --out <폴더>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from egen import store
from preprocessing import clean, grid, message_answers, message_status


def _save(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run(out_dir: Path, db_path: Path | str | None = None, labels_dir: Path | None = None) -> dict:
    with clean.open_readonly(db_path) as conn:
        runs = clean.load_bed_runs(conn)
        beds, bed_stats = clean.clean_beds(conn)
        messages, message_stats = clean.clean_messages(conn)
        hospitals, hospital_stats = clean.clean_hospitals(conn, beds["hpid"])

    beds_grid, grid_stats = grid.build_grid(beds, runs)
    hospitals, quality_stats = grid.hospital_quality(hospitals, beds, beds_grid)

    answers = message_answers.load(labels_dir or out_dir.parent / "labels")
    targets, target_stats = message_status.build_targets(messages, answers)
    message_grid, message_grid_stats = message_status.build_message_grid(targets, beds_grid["t"])

    report = {
        "generated_at": store.iso(store.now_kst()),
        "source_db": str(Path(db_path or store.db_path()).resolve()),
        "beds": bed_stats,
        "messages": message_stats,
        "hospitals": {**hospital_stats, **quality_stats},
        "grid": grid_stats,
        "message_targets": target_stats,
        "message_grid": message_grid_stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _save(beds, out_dir / "beds_clean.parquet")
    _save(beds_grid, out_dir / "beds_grid.parquet")
    _save(messages, out_dir / "messages_clean.parquet")
    _save(hospitals, out_dir / "hospitals_clean.parquet")
    _save(targets, out_dir / "message_targets.parquet")
    _save(message_grid, out_dir / "message_grid.parquet")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=store.DEFAULT_DB.parent / "processed")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
