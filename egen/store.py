"""E-Gen 수집 데이터 저장소 (SQLite).

시각은 모두 KST ISO 문자열 'YYYY-MM-DDTHH:MM:SS+09:00' 로 저장 → 문자열 비교로 시간 순서 비교 가능.

테이블
- hospitals:     기관 기본정보 + 좌표 (하루 1회 갱신)
- bed_snapshots: (hpid, hvidate) 단위 병상 스냅샷. 병원이 갱신하지 않으면 새 행이 생기지 않음.
                 first_seen = 수집기가 처음 본 시각 → 과거 시점 t 에 보였던 정보 = first_seen <= t 인 최신 hvidate.
                 raw 에 원본 전체 필드(JSON) 보관.
- messages:      수용 곤란 메시지. 같은 문구가 질환 코드별로 따로 오므로
                 (hpid, msg_type, sym_code, dept, msg, start_dtm) 단위로 first_seen/last_seen 갱신.
- collection_runs: 수집 성공/실패 기록 (데이터 공백 구간 확인용).
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "egen.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS hospitals (
    hpid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT,
    tel TEXT,
    er_tel TEXT,
    emcls TEXT,
    emcls_name TEXT,
    lat REAL,
    lon REAL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bed_snapshots (
    hpid TEXT NOT NULL,
    hvidate TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    hvec INTEGER,
    raw TEXT NOT NULL,
    PRIMARY KEY (hpid, hvidate)
);
CREATE TABLE IF NOT EXISTS messages (
    hpid TEXT NOT NULL,
    msg_type TEXT NOT NULL,     -- '응급' | '중증'
    sym_code TEXT NOT NULL,     -- symTypCod 질환 코드 (응급 메시지는 빈 값일 수 있음)
    sym_name TEXT,              -- symTypCodMag 질환명
    dept TEXT NOT NULL,         -- trtPrtCodMag 진료과 (없으면 '')
    msg TEXT NOT NULL,
    start_dtm TEXT NOT NULL,
    end_dtm TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    raw TEXT NOT NULL,
    PRIMARY KEY (hpid, msg_type, sym_code, dept, msg, start_dtm)
);
CREATE INDEX IF NOT EXISTS idx_messages_last_seen ON messages(last_seen);
CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    ok INTEGER NOT NULL,
    item_count INTEGER,
    new_count INTEGER,
    error TEXT
);
"""


def now_kst() -> datetime:
    return datetime.now(KST).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.astimezone(KST).replace(microsecond=0).isoformat()


def parse_egen_dtm(value: str | None) -> str | None:
    """'20261030125515' 또는 '202610301255' → KST ISO. 형식이 다르면 None."""
    if not value:
        return None
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M"):
        try:
            return iso(datetime.strptime(value, fmt).replace(tzinfo=KST))
        except ValueError:
            continue
    return None


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def db_path() -> Path:
    return Path(os.getenv("EGEN_DB_PATH") or DEFAULT_DB)


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # 수집기 쓰기 중에도 서버가 읽을 수 있게
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------- 쓰기 (수집기) ----------

def save_hospitals(conn: sqlite3.Connection, items: list[dict], seen_at: str) -> int:
    rows = [
        (it["hpid"], it.get("dutyName", ""), it.get("dutyAddr"), it.get("dutyTel1"), it.get("dutyTel3"),
         it.get("dutyEmcls"), it.get("dutyEmclsName"),
         _to_float(it.get("wgs84Lat")), _to_float(it.get("wgs84Lon")), seen_at)
        for it in items if it.get("hpid")
    ]
    conn.executemany(
        """INSERT INTO hospitals VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(hpid) DO UPDATE SET name=excluded.name, address=excluded.address, tel=excluded.tel,
             er_tel=excluded.er_tel, emcls=excluded.emcls, emcls_name=excluded.emcls_name,
             lat=excluded.lat, lon=excluded.lon, updated_at=excluded.updated_at""",
        rows,
    )
    return len(rows)


def save_bed_snapshots(conn: sqlite3.Connection, items: list[dict], seen_at: str) -> int:
    """새로 갱신된 (hpid, hvidate) 만 추가. 추가된 행 수 반환."""
    before = conn.total_changes
    for it in items:
        hvidate = parse_egen_dtm(it.get("hvidate"))
        if not it.get("hpid") or not hvidate:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO bed_snapshots VALUES (?,?,?,?,?)",
            (it["hpid"], hvidate, seen_at, _to_int(it.get("hvec")), json.dumps(it, ensure_ascii=False)),
        )
    return conn.total_changes - before


def save_messages(conn: sqlite3.Connection, items: list[dict], seen_at: str) -> int:
    """메시지 upsert. 새로 등장한 메시지 수 반환."""
    new = 0
    for it in items:
        if not it.get("hpid") or not it.get("symBlkMsg"):
            continue
        hpid, msg_type, sym_code = it["hpid"], it.get("symBlkMsgTyp", ""), it.get("symTypCod", "")
        dept, msg = it.get("trtPrtCodMag", ""), it["symBlkMsg"]
        start = parse_egen_dtm(it.get("symBlkSttDtm")) or ""
        end = parse_egen_dtm(it.get("symBlkEndDtm"))
        cur = conn.execute(
            """UPDATE messages SET last_seen=?, end_dtm=?
               WHERE hpid=? AND msg_type=? AND sym_code=? AND dept=? AND msg=? AND start_dtm=?""",
            (seen_at, end, hpid, msg_type, sym_code, dept, msg, start),
        )
        if cur.rowcount == 0:
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (hpid, msg_type, sym_code, it.get("symTypCodMag") or None, dept,
                 msg, start, end, seen_at, seen_at, json.dumps(it, ensure_ascii=False)),
            )
            new += 1
    return new


def log_run(conn: sqlite3.Connection, started_at: str, kind: str, ok: bool,
            item_count: int | None = None, new_count: int | None = None, error: str | None = None) -> None:
    conn.execute(
        "INSERT INTO collection_runs (started_at, kind, ok, item_count, new_count, error) VALUES (?,?,?,?,?,?)",
        (started_at, kind, int(ok), item_count, new_count, error),
    )


# ---------- 읽기 (서버/알고리즘) ----------

def latest_success(conn: sqlite3.Connection, kind: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(started_at) FROM collection_runs WHERE kind=? AND ok=1", (kind,)
    ).fetchone()
    return row[0]


def current_hospital_states(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """좌표가 있고 병상 정보가 있는 기관의 최신 상태."""
    return conn.execute(
        """SELECT h.hpid, h.name, h.address, h.tel, h.er_tel, h.lat, h.lon, b.hvidate, b.hvec
           FROM hospitals h
           JOIN bed_snapshots b ON b.hpid = h.hpid
           WHERE h.lat IS NOT NULL AND h.lon IS NOT NULL
             AND b.hvidate = (SELECT MAX(hvidate) FROM bed_snapshots WHERE hpid = h.hpid)"""
    ).fetchall()


def active_messages(conn: sqlite3.Connection, hpids: list[str]) -> dict[str, list[str]]:
    """가장 최근 메시지 수집에서 보였고 아직 종료되지 않은 메시지.

    같은 문구는 질환명을 묶어 '[대동맥응급(복부), 수족지접합] 의료진 부족' 형태로 합친다.
    """
    last = latest_success(conn, "messages")
    if not last or not hpids:
        return {}
    marks = ",".join("?" * len(hpids))
    rows = conn.execute(
        f"""SELECT hpid, msg, sym_name FROM messages
            WHERE last_seen = ? AND (end_dtm IS NULL OR end_dtm >= ?) AND hpid IN ({marks})
            ORDER BY start_dtm DESC, sym_code""",
        (last, iso(now_kst()), *hpids),
    ).fetchall()
    grouped: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        names = grouped.setdefault(r["hpid"], {}).setdefault(r["msg"], [])
        # '응급실'(Y000) 은 특정 질환이 아닌 일반 메시지라 꼬리표에서 제외
        if r["sym_name"] and r["sym_name"] != "응급실" and r["sym_name"] not in names:
            names.append(r["sym_name"])
    return {
        hpid: [f"[{', '.join(names)}] {msg}" if names else msg for msg, names in msgs.items()]
        for hpid, msgs in grouped.items()
    }


def hospital_detail(conn: sqlite3.Connection, hpid: str) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT h.*, b.hvidate, b.hvec FROM hospitals h
           LEFT JOIN bed_snapshots b ON b.hpid = h.hpid
             AND b.hvidate = (SELECT MAX(hvidate) FROM bed_snapshots WHERE hpid = h.hpid)
           WHERE h.hpid = ?""",
        (hpid,),
    ).fetchone()
