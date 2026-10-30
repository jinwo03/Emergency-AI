"""E-Gen 주기 수집기.

실행:
    python -m egen.collector          # 5분 주기로 계속 수집 (Ctrl+C 종료)
    python -m egen.collector --once   # 한 번만 수집하고 종료

- 병상/메시지: 5분마다 (정각 기준 00, 05, 10 ... 분에 맞춰 실행)
- 기관 목록(좌표): 시작 시 + 24시간마다
- 실패해도 종료하지 않고 collection_runs 에 기록 후 다음 주기에 재시도
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from egen import client, store

INTERVAL_SEC = 300
HOSPITAL_REFRESH = timedelta(hours=24)

log = logging.getLogger("collector")

JOBS = {
    "hospitals": (client.fetch_hospitals, store.save_hospitals),
    "beds": (client.fetch_realtime_beds, store.save_bed_snapshots),
    "messages": (client.fetch_messages, store.save_messages),
}


def run_job(kind: str) -> bool:
    fetch, save = JOBS[kind]
    started = store.iso(store.now_kst())
    try:
        items = fetch()
    except Exception as e:  # 네트워크/인증/파싱 오류 모두 기록만 하고 계속
        log.error("%s 수집 실패: %s", kind, e)
        with store.connect() as conn:
            store.log_run(conn, started, kind, ok=False, error=str(e)[:500])
        return False

    with store.connect() as conn:
        result = save(conn, items, started)
        new = None if kind == "hospitals" else result
        store.log_run(conn, started, kind, ok=True, item_count=len(items), new_count=new)
    log.info("%s: %d건 수신%s", kind, len(items), "" if new is None else f", 신규 {new}건")
    return True


def hospitals_due() -> bool:
    with store.connect() as conn:
        last = store.latest_success(conn, "hospitals")
    if last is None:
        return True
    return store.now_kst() - datetime.fromisoformat(last) >= HOSPITAL_REFRESH


def collect_once() -> None:
    if hospitals_due():
        run_job("hospitals")
    run_job("beds")
    run_job("messages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="한 번만 수집하고 종료")
    args = parser.parse_args()

    log_file = Path(store.db_path()).parent / "collector.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )
    log.info("DB: %s", store.db_path())

    if args.once:
        collect_once()
        return

    while True:
        collect_once()
        # 다음 5분 경계까지 대기
        time.sleep(INTERVAL_SEC - (time.time() % INTERVAL_SEC))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("수집기 종료")
