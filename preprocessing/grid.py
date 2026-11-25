"""전처리 2단계: 병원마다 제각각인 갱신 시점을 5분 간격 격자로 맞춘다.

격자 시각 t 의 값은 first_seen <= t 인 기록 중 가장 최근 것이다. 즉 그 시점에 실제로 알 수 있었던 값만 쓴다.
병원 갱신 시각(hvidate) 기준으로 30분이 지난 값은 결측으로 바꾸고 is_stale 로 표시한다.
"""
import pandas as pd

from preprocessing.clean import MEASURE_COLUMNS

SLOT = pd.Timedelta(minutes=5)
MAX_DATA_AGE_MIN = 30
MIN_FRESH_RATIO = 0.5


def build_grid(beds: pd.DataFrame, runs: pd.Series) -> tuple[pd.DataFrame, dict]:
    slots = pd.date_range(runs.iloc[0].ceil(SLOT), runs.iloc[-1].floor(SLOT), freq=SLOT, name="t").as_unit("ns")

    # 각 병원은 처음 관측된 시점부터 격자에 들어간다
    first_seen = beds.groupby("hpid")["first_seen"].min().rename("hospital_first_seen").reset_index()
    frame = first_seen.merge(pd.DataFrame({"t": slots}), how="cross")
    frame = frame[frame["t"] >= frame["hospital_first_seen"]].drop(columns="hospital_first_seen")

    grid = pd.merge_asof(
        frame.sort_values("t"),
        beds.sort_values("first_seen"),
        left_on="t",
        right_on="first_seen",
        by="hpid",
        direction="backward",
    )

    grid["data_age_min"] = ((grid["t"] - grid["hvidate"]).dt.total_seconds() / 60).clip(lower=0)
    grid["is_stale"] = grid["data_age_min"] > MAX_DATA_AGE_MIN
    for column in [*MEASURE_COLUMNS, "hvec_suspect"]:
        grid[column] = grid[column].mask(grid["is_stale"])

    last_run = pd.merge_asof(
        pd.DataFrame({"t": slots}),
        pd.DataFrame({"run_at": runs}),
        left_on="t",
        right_on="run_at",
        direction="backward",
    )
    gap_slots = last_run.loc[last_run["t"] - last_run["run_at"] >= SLOT, "t"]
    grid["collection_gap"] = grid["t"].isin(gap_slots)

    grid = grid.sort_values(["hpid", "t"]).reset_index(drop=True)
    return grid, {
        "rows": len(grid),
        "hospitals": int(grid["hpid"].nunique()),
        "slots": len(slots),
        "start": slots[0].isoformat(),
        "end": slots[-1].isoformat(),
        "stale_rows": int(grid["is_stale"].sum()),
        "gap_slots": len(gap_slots),
    }


def hospital_quality(hospitals: pd.DataFrame, beds: pd.DataFrame, grid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """병원을 지우지 않고 학습 제외 후보만 표시한다.

    전체 기간 기준 값이므로, 학습 대상을 확정할 때는 학습 구간으로 다시 계산한다.
    """
    quality = pd.DataFrame({
        "snapshots": beds.groupby("hpid").size(),
        "constant_hvec": beds.groupby("hpid")["hvec"].nunique() <= 1,
        "fresh_ratio": 1 - grid.groupby("hpid")["is_stale"].mean(),
    }).reset_index()
    quality["low_data"] = quality["fresh_ratio"] < MIN_FRESH_RATIO
    quality["exclude_candidate"] = quality["constant_hvec"] | quality["low_data"]
    merged = hospitals.merge(quality, on="hpid", how="inner")
    return merged, {
        "constant_hvec": int(merged["constant_hvec"].sum()),
        "low_data": int(merged["low_data"].sum()),
        "exclude_candidate": int(merged["exclude_candidate"].sum()),
    }
