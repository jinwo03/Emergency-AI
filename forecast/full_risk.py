"""도착 시점 만실 위험: 병원 h 에 시각 t 에 출발하면 t + H 분에 응급실 가용 병상(hvec)이 0 이하일 확률.

실행:
    python -m forecast.full_risk                       # 30·60분, 구간 이동 평가. 결과: data/models/full_risk_report.json
    python -m forecast.full_risk --horizons 30 --folds 4 --fold-days 2 --seeds 0 1 2

설계
- 표본: 병원 × 5분 격자 시각. t 와 t+H 둘 다 정상값(오래된 값·수집 공백·의심값 아님)인 칸만 쓴다.
- 특징은 모두 t 시점에 알 수 있는 값이다. 병원 평소 패턴(시간대별 평균)은 학습 기간으로만 만든다.
- 구간 이동 평가(rolling origin): 마지막 folds × fold_days 일을 fold_days 일씩 차례로 시험 구간으로 쓴다.
  구간마다 그 앞 VALID_DAYS 일은 조기 종료용, 그보다 앞은 전부 학습이다. 병원 평소 패턴도 구간마다 학습 기간으로만 만든다.
- LightGBM 은 seed 여러 개로 학습해 확률을 평균한다. 지표는 구간별 값과 평균·표준편차를 낸다.
- 차이의 불확실성: 모든 시험 구간 예측을 모아 병원 단위로 다시 뽑아(bootstrap) 95% 구간을 낸다.
- 비교: 지금 값 그대로(만실이면 1, 아니면 0) / 현재 병상 수별 확률표 / LightGBM (특징 묶음을 하나씩 더함).
- 평가: Brier, 로그 손실, AUC, 평균 정밀도(AP), 보정 오차(ECE). 전체와 '거의 참(지금 1~3자리)' 구간을 따로 본다.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from egen import store

SLOT_MIN = 5
VALID_DAYS = 2
NEAR_FULL = (1, 3)
NEIGHBORS = 5
LAGS_MIN = [5, 15, 30, 60, 120]

FEATURE_SETS = {
    "base": ["hvec", "hvs01", "occupancy", *[f"d{m}" for m in LAGS_MIN], "min60", "max60", "data_age_min",
             "hour", "dow", "weekend", "target_hour", "emcls", "hosp_mean", "prof_now", "prof_target", "prof_delta"],
    "spatial": ["nb_full", "nb_occ", "nb_docc30"],
    "message": ["msg_blocked", "msg_all_blocked", "msg_ask", "severe_blocked", "msg_series"],
}


# ---------- 데이터 ----------

def _matrix(grid: pd.DataFrame, column: str) -> pd.DataFrame:
    return grid.pivot(index="t", columns="hpid", values=column)


def _neighbors(hospitals: pd.DataFrame, columns: pd.Index, k: int = NEIGHBORS) -> dict[str, list[str]]:
    """위경도 기준 가까운 병원 k 곳 (자기 자신 제외)."""
    h = hospitals.set_index("hpid").reindex(columns)
    lat, lon = np.radians(h["lat"].to_numpy(float)), np.radians(h["lon"].to_numpy(float))
    d = (np.sin((lat[:, None] - lat[None]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2)
    np.fill_diagonal(d, np.inf)
    d = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d, axis=1)[:, :k]
    return {columns[i]: [columns[j] for j in order[i]] for i in range(len(columns))}


def _message_features(messages: pd.DataFrame, index: pd.Index, columns: pd.Index) -> dict[str, pd.DataFrame]:
    m = messages
    general, severe = m["msg_type"] == "응급", m["msg_type"] == "중증"
    blocked = m["status"].isin(["불가", "기존 환자만"])
    whole = (m["departments"] == "전체") & (m["disease_codes"] == "") & (m["conditions"] == "")
    flags = {
        "msg_blocked": general & blocked,
        "msg_all_blocked": general & blocked & whole,
        "msg_ask": general & (m["caution"].str.contains("문의", na=False) | (m["status"] == "어려움")),
        "severe_blocked": severe & blocked,
    }
    out = {}
    for name, mask in flags.items():
        counts = m.loc[mask].groupby(["t", "hpid"]).size().unstack(fill_value=0)
        out[name] = counts.reindex(index=index, columns=columns, fill_value=0).astype(float)
    series = m.loc[general].groupby(["t", "hpid"])["series_id"].nunique().unstack(fill_value=0)
    out["msg_series"] = series.reindex(index=index, columns=columns, fill_value=0).astype(float)
    return out


def build_dataset(processed: Path, horizon: int) -> pd.DataFrame:
    """병원 × 시각 표본과 특징, 목표(full_future). 학습 기간에 따라 달라지는 특징은 add_profile 이 붙인다."""
    grid = pd.read_parquet(processed / "beds_grid.parquet",
                           columns=["hpid", "t", "hvec", "hvs01", "data_age_min", "is_stale", "collection_gap",
                                    "hvec_suspect"])
    hospitals = pd.read_parquet(processed / "hospitals_clean.parquet")
    grid = grid[grid["hpid"].isin(hospitals.loc[~hospitals["exclude_candidate"], "hpid"])]
    grid["ok"] = ~grid["is_stale"] & ~grid["collection_gap"] & (grid["hvec_suspect"] != True) & grid["hvec"].notna()

    hvec = _matrix(grid, "hvec").astype(float)
    T, H = hvec.index, hvec.columns
    ok = _matrix(grid, "ok").reindex(index=T, columns=H).fillna(False).astype(bool)
    hvec = hvec.where(ok)
    cap = _matrix(grid, "hvs01").astype(float).reindex(index=T, columns=H)
    cap = cap.where(cap > 0).ffill()
    age = _matrix(grid, "data_age_min").astype(float).reindex(index=T, columns=H)

    k = horizon // SLOT_MIN

    f: dict[str, pd.DataFrame] = {"hvec": hvec, "hvs01": cap, "occupancy": 1 - hvec / cap, "data_age_min": age}
    for lag in LAGS_MIN:
        f[f"d{lag}"] = hvec - hvec.shift(lag // SLOT_MIN)
    f["min60"] = hvec.rolling(60 // SLOT_MIN, min_periods=1).min()
    f["max60"] = hvec.rolling(60 // SLOT_MIN, min_periods=1).max()

    def broadcast(values) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(np.asarray(values, float)[:, None], len(H), axis=1), index=T, columns=H)

    target_time = T + pd.Timedelta(minutes=horizon)
    f["hour"] = broadcast(T.hour + T.minute / 60)
    f["dow"] = broadcast(T.day_of_week)
    f["weekend"] = broadcast(T.day_of_week >= 5)
    f["target_hour"] = broadcast(target_time.hour + target_time.minute / 60)
    emcls = hospitals.set_index("hpid")["emcls"].reindex(H).astype("category").cat.codes
    f["emcls"] = pd.DataFrame(np.repeat(emcls.to_numpy()[None], len(T), axis=0), index=T, columns=H)

    # 주변 병원: 만실 비율, 평균 점유율, 30분 점유율 변화
    nb = _neighbors(hospitals, H)
    full_now = (hvec <= 0).astype(float).where(hvec.notna())
    occ = f["occupancy"]
    f["nb_full"] = pd.DataFrame({h: full_now[nb[h]].mean(axis=1) for h in H})
    f["nb_occ"] = pd.DataFrame({h: occ[nb[h]].mean(axis=1) for h in H})
    f["nb_docc30"] = f["nb_occ"] - f["nb_occ"].shift(30 // SLOT_MIN)

    messages = pd.read_parquet(processed / "message_grid.parquet",
                               columns=["hpid", "t", "series_id", "msg_type", "departments", "disease_codes",
                                        "conditions", "status", "caution"])
    f |= _message_features(messages, T, H)

    future = hvec.shift(-k)
    mask = (hvec.notna() & future.notna()).to_numpy()
    rows = {name: frame.reindex(index=T, columns=H).to_numpy(float)[mask] for name, frame in f.items()}
    tt = np.broadcast_to(np.asarray(T)[:, None], mask.shape)[mask]
    hh = np.broadcast_to(np.asarray(H)[None], mask.shape)[mask]
    df = pd.DataFrame({"t": tt, "hpid": hh, **rows})
    df["full_future"] = (future.to_numpy()[mask] <= 0).astype(int)
    t_index = pd.DatetimeIndex(df["t"])
    df["day"] = t_index.normalize()
    df["hour_int"] = t_index.hour
    df["target_hour_int"] = (t_index + pd.Timedelta(minutes=horizon)).hour
    return df


def add_profile(df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """병원 평균 병상과 병원 × 시간대별 평균을 train_end 이전 행으로만 만들어 붙인다."""
    train = df[df["t"] < train_end]
    out = df.copy()
    out["hosp_mean"] = out["hpid"].map(train.groupby("hpid")["hvec"].mean())
    prof = train.groupby(["hpid", "hour_int"])["hvec"].mean()
    out["prof_now"] = pd.MultiIndex.from_arrays([out["hpid"], out["hour_int"]]).map(prof).to_numpy(float)
    out["prof_target"] = pd.MultiIndex.from_arrays([out["hpid"], out["target_hour_int"]]).map(prof).to_numpy(float)
    out["prof_delta"] = out["prof_target"] - out["prof_now"]
    return out


def folds(df: pd.DataFrame, n_folds: int, fold_days: int) -> list[dict]:
    """마지막 n_folds × fold_days 일을 차례로 시험 구간으로. 각 구간: 학습 끝(=검증 시작), 시험 시작, 시험 끝."""
    last_day = df["day"].max()
    out = []
    for i in range(n_folds):
        test_start = last_day - pd.Timedelta(days=(n_folds - i) * fold_days - 1)
        out.append({"train_end": test_start - pd.Timedelta(days=VALID_DAYS), "test_start": test_start,
                    "test_end": test_start + pd.Timedelta(days=fold_days)})
    return out


# ---------- 모델 ----------

def persistence(df: pd.DataFrame) -> np.ndarray:
    return (df["hvec"] <= 0).astype(float).to_numpy()


def table_by_current(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """현재 병상 수(-5~20으로 자름)별 학습 기간 만실 비율."""
    clip = lambda s: s.clip(-5, 20).round()
    rate = train.groupby(clip(train["hvec"]))["full_future"].mean()
    return clip(test["hvec"]).map(rate).fillna(train["full_future"].mean()).to_numpy()


def fit_lgbm(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int = 0) -> lgb.Booster:
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed, "verbose": -1}
    dtrain = lgb.Dataset(train[features], train["full_future"], categorical_feature=["emcls"] if "emcls" in features else "auto")
    dvalid = lgb.Dataset(valid[features], valid["full_future"], reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


# ---------- 평가 ----------

def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
           "brier": round(float(brier_score_loss(y, p)), 5), "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5)}
    if 0 < y.sum() < len(y):
        out |= {"auc": round(float(roc_auc_score(y, p)), 4), "ap": round(float(average_precision_score(y, p)), 4)}
    out["ece"] = round(ece(y, p), 4)
    return out


SUBSETS = {
    "all": lambda d: np.ones(len(d), bool),
    "available_now": lambda d: (d["hvec"] > 0).to_numpy(),
    "near_full_1_3": lambda d: d["hvec"].between(*NEAR_FULL).to_numpy(),
}
KEYS = ["t", "hpid", "hvec", "full_future", "fold"]


def predict_fold(df: pd.DataFrame, fold: dict, seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    """한 시험 구간의 모델별 예측 확률과 특징 중요도 (seed 평균)."""
    data = add_profile(df, fold["train_end"])
    train = data[data["t"] < fold["train_end"]]
    valid = data[(data["t"] >= fold["train_end"]) & (data["t"] < fold["test_start"])]
    test = data[(data["t"] >= fold["test_start"]) & (data["t"] < fold["test_end"])]
    out = test[["t", "hpid", "hvec", "full_future"]].copy()
    out["persistence"] = persistence(test)
    out["table_by_current"] = table_by_current(pd.concat([train, valid]), test)
    importance, used = {}, []
    for i, features in enumerate(FEATURE_SETS.values()):
        used = used + features
        label = "lgbm_" + "+".join(list(FEATURE_SETS)[:i + 1])
        probs, gains = [], np.zeros(len(used))
        for seed in seeds:
            model = fit_lgbm(train, valid, used, seed)
            probs.append(model.predict(test[used], num_iteration=model.best_iteration))
            gains += model.feature_importance("gain")
        out[label] = np.mean(probs, axis=0)
        importance[label] = dict(sorted(zip(used, (gains / len(seeds)).round(0).tolist()), key=lambda x: -x[1])[:10])
    return out, importance


def bootstrap_diff(pred: pd.DataFrame, a: str, b: str, metric: str, reps: int = 200, seed: int = 0) -> dict:
    """병원 단위로 다시 뽑아 지표 차이(a - b)의 95% 구간을 낸다."""
    fn = {"auc": roc_auc_score, "ap": average_precision_score, "brier": brier_score_loss}[metric]
    groups = pred.groupby("hpid").indices
    keys = np.array(list(groups))
    y, pa, pb = pred["full_future"].to_numpy(), pred[a].to_numpy(), pred[b].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([groups[h] for h in rng.choice(keys, len(keys))])
        if 0 < y[idx].sum() < len(idx):
            diffs.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float(fn(y, pa) - fn(y, pb)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def summarize(pred: pd.DataFrame, models: list[str], select) -> dict:
    per_fold = {m: [] for m in models}
    for _, part in pred.groupby("fold"):
        sel = select(part)
        for m in models:
            per_fold[m].append(metrics(part["full_future"].to_numpy()[sel], part[m].to_numpy()[sel]))
    out = {}
    for m, values in per_fold.items():
        keys = [k for k in ("brier", "log_loss", "auc", "ap", "ece") if all(k in v for v in values)]
        out[m] = {k: {"mean": round(float(np.mean([v[k] for v in values])), 4),
                      "std": round(float(np.std([v[k] for v in values])), 4),
                      "per_fold": [v[k] for v in values]} for k in keys}
        out[m] |= {"n": sum(v["n"] for v in values),
                   "positive_rate": round(float(np.mean([v["positive_rate"] for v in values])), 4)}
    return out


def evaluate(df: pd.DataFrame, n_folds: int, fold_days: int, seeds: list[int]) -> tuple[dict, pd.DataFrame]:
    preds, info, importance = [], [], {}
    for i, fold in enumerate(folds(df, n_folds, fold_days)):
        pred, importance[f"fold{i}"] = predict_fold(df, fold, seeds)
        pred["fold"] = i
        preds.append(pred)
        info.append({k: str(v.date()) for k, v in fold.items()} | {"test_rows": len(pred)})
        print(f"  구간 {i + 1}/{n_folds} 완료", flush=True)
    pred = pd.concat(preds, ignore_index=True)
    models = [c for c in pred.columns if c not in KEYS]
    report = {"folds": info, "seeds": seeds}
    report |= {subset: summarize(pred, models, select) for subset, select in SUBSETS.items()}
    near = pred[SUBSETS["near_full_1_3"](pred)]
    pairs = [("lgbm_base", "table_by_current")] + [(m, "lgbm_base") for m in models
                                                   if m.startswith("lgbm_") and m != "lgbm_base"]
    report["near_full_1_3_differences"] = {f"{a} - {b}": {k: bootstrap_diff(near, a, b, k) for k in ("auc", "ap", "brier")}
                                           for a, b in pairs}
    report["feature_importance_gain"] = importance
    return report, pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=store.DEFAULT_DB.parent / "processed")
    parser.add_argument("--out", type=Path, default=store.DEFAULT_DB.parent / "models" / "full_risk_report.json")
    parser.add_argument("--horizons", type=int, nargs="+", default=[30, 60])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold-days", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    report = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for horizon in args.horizons:
        df = build_dataset(args.processed, horizon)
        report[f"{horizon}min"], pred = evaluate(df, args.folds, args.fold_days, args.seeds)
        pred.to_parquet(args.out.with_name(f"full_risk_predictions_{horizon}min.parquet"), index=False)
        print(f"{horizon}분 완료", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
