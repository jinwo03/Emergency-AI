"""전처리 1단계: 수집 DB 를 읽어 학습에 쓸 수 있게 정리한다.

원본 DB 는 읽기 전용으로만 연다. 수집기가 도는 중에 실행해도 되고, 결과는 data/processed/ 에만 쓴다.

정리 규칙
- 병상 수 음수는 과밀 정보이므로 자르지 않는다. 값이 없는 행만 제외한다.
- 병원이 보고하지 않은 항목은 0 으로 채우지 않고 결측으로 둔다 (*_reported 로 구분).
- 장비 항목은 Y/N/N1 세 값을 쓴다. 가용 여부(*_available)와 원본 값(*_status)을 함께 보관한다.
- '불가능메세지등록' 은 일반 메시지(Y000)일 때만 제외한다. 중증 메시지는 질환 코드 자체가 불가 정보다.
- 메시지 원문에 섞인 환자번호·이름·의료진 이름·전화번호는 가린다. 원문은 원본 DB 에만 남는다.
"""
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from egen import store

PLACEHOLDER_MSG = "불가능메세지등록"
GENERAL_SYM_CODE = "Y000"

# 응급실 기준 병상(hvs01)과 병상 항목
BED_FIELDS = ["hvs01", "hvoc", "hvcc", "hvncc", "hvccc", "hvicc", "hvgc"]
EQUIPMENT_FIELDS = {
    "hvctayn": "ct",
    "hvmriayn": "mri",
    "hvangioayn": "angio",
    "hvventiayn": "vent",
    "hvventisoayn": "vent_infant",
    "hvincuayn": "incubator",
    "hvcrrtayn": "crrt",
    "hvecmoayn": "ecmo",
    "hvhypoayn": "hypothermia",
    "hvoxyayn": "hyperbaric_oxygen",
    "hvamyn": "ambulance",
}
PERSONAL_INFO_PATTERNS = [
    (re.compile(r"\d{5,}\s*[가-힣][Oo○*＊xX][가-힣]?"), "[환자정보]"),
    (re.compile(r"(?<![가-힣])[가-힣][Oo○*＊xX][가-힣](?![가-힣])"), "[환자정보]"),
    (re.compile(r"(?i)(?<![a-z])pf\.?\s*[가-힣]{2,4}"), "[의료진]"),
    (re.compile(r"(?i)(?<![a-z])dr\.?\s*[가-힣]{2,4}"), "[의료진]"),
    (re.compile(r"[가-힣]{2,4}\s*교수"), "[의료진]"),
    (re.compile(r"\d{2,4}-\d{3,4}-\d{4}"), "[전화번호]"),
    (re.compile(r"(?<![\d-])\d{5,}(?![\d-])"), "[번호]"),
]
MEASURE_COLUMNS = (
    ["hvec", *BED_FIELDS]
    + [f"{name}_status" for name in EQUIPMENT_FIELDS.values()]
    + [f"{name}_available" for name in EQUIPMENT_FIELDS.values()]
)


@contextmanager
def open_readonly(path: Path | str | None = None):
    """원본 DB 를 쓰기 불가능한 연결로 연다. 전체 읽기가 같은 시점을 보도록 트랜잭션으로 묶는다."""
    uri = f"{Path(path or store.db_path()).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        yield conn
    finally:
        conn.close()


def _timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="ISO8601").dt.as_unit("ns")


def _read(conn: sqlite3.Connection, sql: str, *params) -> pd.DataFrame:
    cur = conn.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])


def load_bed_runs(conn: sqlite3.Connection) -> pd.Series:
    """병상 수집에 성공한 시각. 격자의 범위와 수집 공백 판단에 쓴다."""
    runs = _read(conn, "SELECT started_at FROM collection_runs WHERE kind='beds' AND ok=1 ORDER BY started_at")
    if runs.empty:
        raise ValueError("병상 수집 성공 기록이 없습니다")
    return _timestamps(runs["started_at"]).sort_values().reset_index(drop=True)


def clean_beds(conn: sqlite3.Connection) -> tuple[pd.DataFrame, dict]:
    fields = [*BED_FIELDS, *EQUIPMENT_FIELDS]
    extracted = ", ".join(f"json_extract(raw, '$.{f}') AS {f}" for f in fields)
    df = _read(conn, f"SELECT hpid, hvidate, first_seen, hvec, {extracted} FROM bed_snapshots")
    stats = {"source_rows": len(df), "null_hvec_dropped": int(df["hvec"].isna().sum())}

    df = df.dropna(subset=["hvec"])
    df["hvidate"] = _timestamps(df["hvidate"])
    df["first_seen"] = _timestamps(df["first_seen"])
    df = df.sort_values(["hpid", "first_seen", "hvidate"])

    # 더 최근 갱신을 이미 본 뒤에 뒤늦게 나타난 과거 기록은 그 시점의 최신값이 아니다
    latest = df.groupby("hpid")["hvidate"].cummax()
    stats["out_of_order_dropped"] = int((df["hvidate"] < latest).sum())
    df = df[df["hvidate"] >= latest].copy()

    df["hvec"] = df["hvec"].astype("Int64")
    for field in BED_FIELDS:
        df[f"{field}_reported"] = df[field].notna()
        df[field] = pd.to_numeric(df[field], errors="coerce").astype("Float64")
    for field, name in EQUIPMENT_FIELDS.items():
        df[f"{name}_status"] = df.pop(field)
        df[f"{name}_available"] = df[f"{name}_status"].map({"Y": True, "N": False, "N1": False}).astype("boolean")

    # 값은 유지하고 표시만 한다: 기준 병상보다 많이 비었거나, 기준 병상의 2배를 넘게 과밀한 경우
    base = df["hvs01"]
    df["hvec_suspect"] = ((df["hvec"] > base) | (df["hvec"] < -base)).fillna(False).astype(bool)

    stats.update(
        rows=len(df),
        hospitals=int(df["hpid"].nunique()),
        negative_hvec=int((df["hvec"] < 0).sum()),
        zero_hvec=int((df["hvec"] == 0).sum()),
        suspect_hvec=int(df["hvec_suspect"].sum()),
        missing_hvs01=int(df["hvs01"].isna().sum()),
        hvidate_after_first_seen=int((df["hvidate"] > df["first_seen"]).sum()),
    )
    return df.reset_index(drop=True), stats


def mask_personal_info(text: str) -> str:
    for pattern, replacement in PERSONAL_INFO_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_message(text: str) -> str:
    """같은 문구 판별용. 공백·대소문자·끝 마침표 차이만 없앤다. 원문은 별도 컬럼에 보관한다."""
    return "".join(text.split()).rstrip(".").lower()


def clean_messages(conn: sqlite3.Connection) -> tuple[pd.DataFrame, dict]:
    df = _read(
        conn,
        """SELECT hpid, msg_type, sym_code, sym_name, dept, msg, start_dtm, end_dtm, first_seen, last_seen
           FROM messages""",
    )
    is_placeholder = df["msg"] == PLACEHOLDER_MSG
    no_information = is_placeholder & (df["sym_code"] == GENERAL_SYM_CODE)
    stats = {
        "source_rows": len(df),
        "placeholder_rows": int(is_placeholder.sum()),
        "placeholder_dropped": int(no_information.sum()),
        "placeholder_kept_with_disease_code": int((is_placeholder & ~no_information).sum()),
    }

    # 수집 시작 때 이미 떠 있었거나 지금도 떠 있으면 실제 지속 시간을 알 수 없다
    df["left_censored"] = df["first_seen"] == df["first_seen"].min()
    df["right_censored"] = df["last_seen"] == df["last_seen"].max()
    df["is_placeholder"] = is_placeholder
    df = df[~no_information].copy()

    masked = df["msg"].map(mask_personal_info)
    stats["personal_info_masked_rows"] = int((masked != df["msg"]).sum())
    df["msg"] = masked

    df["msg_norm"] = df["msg"].map(normalize_message)
    for column in ("start_dtm", "end_dtm", "first_seen", "last_seen"):
        df[column] = _timestamps(df[column])

    # 야간마다 다시 올리는 것처럼 같은 메시지가 반복 등록되므로 묶음 안에서 회차를 매긴다
    series = ["hpid", "msg_type", "sym_code", "dept", "msg"]
    df = df.sort_values([*series, "start_dtm"])
    df["series_id"] = df.groupby(series, sort=False).ngroup()
    df["episode_no"] = df.groupby("series_id").cumcount() + 1

    repeats = df.groupby("series_id")["episode_no"].transform("max")
    stats.update(
        rows=len(df),
        unique_messages=int(df["msg"].nunique()),
        unique_normalized_messages=int(df["msg_norm"].nunique()),
        series=int(df["series_id"].nunique()),
        repeated_series=int(df.loc[repeats > 1, "series_id"].nunique()),
        repeated_series_rows=int((repeats > 1).sum()),
    )
    return df.reset_index(drop=True), stats


def clean_hospitals(conn: sqlite3.Connection, bed_hpids: pd.Series) -> tuple[pd.DataFrame, dict]:
    """병상 데이터가 있는 기관만 남긴다."""
    df = _read(conn, "SELECT hpid, name, address, emcls, emcls_name, lat, lon FROM hospitals")
    has_beds = df["hpid"].isin(bed_hpids)
    return df[has_beds].reset_index(drop=True), {
        "source_rows": len(df),
        "without_bed_data_dropped": int((~has_beds).sum()),
        "rows": int(has_beds.sum()),
    }
