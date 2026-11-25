import numpy as np
import pandas as pd

from forecast import full_risk


def frame(days=10, hospitals=("A", "B")) -> pd.DataFrame:
    t = pd.date_range("2026-11-01", periods=days * 24, freq="h", tz="Asia/Seoul")
    rows = []
    for h in hospitals:
        for x in t:
            rows.append({"t": x, "hpid": h, "hvec": float(x.day), "full_future": int(x.hour == 14)})
    df = pd.DataFrame(rows)
    df["day"] = df["t"].dt.normalize()
    df["hour_int"] = df["t"].dt.hour
    df["target_hour_int"] = (df["t"] + pd.Timedelta(minutes=30)).dt.hour
    return df


def test_folds_cover_last_days_in_order():
    df = frame(days=10)
    fs = full_risk.folds(df, n_folds=2, fold_days=2)
    last = df["day"].max()
    assert fs[-1]["test_end"] == last + pd.Timedelta(days=1)
    assert fs[0]["test_end"] == fs[1]["test_start"]
    assert all(f["train_end"] == f["test_start"] - pd.Timedelta(days=full_risk.VALID_DAYS) for f in fs)


def test_profile_uses_only_training_period():
    df = frame(days=10)
    train_end = pd.Timestamp("2026-11-04", tz="Asia/Seoul")   # 1~3일만 학습
    out = full_risk.add_profile(df, train_end)
    assert set(out["hosp_mean"]) == {2.0}                    # (1+2+3)/3, 시험 기간 값이 섞이지 않음
    assert set(out["prof_now"]) == {2.0}


def test_table_by_current_uses_training_rates():
    train = pd.DataFrame({"hvec": [1, 1, 1, 1, 5, 5], "full_future": [1, 1, 0, 0, 0, 0]})
    test = pd.DataFrame({"hvec": [1, 5, 30]})
    np.testing.assert_allclose(full_risk.table_by_current(train, test), [0.5, 0.0, 2 / 6])


def test_metrics_and_bootstrap_on_perfect_vs_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    pred = pd.DataFrame({"hpid": rng.choice(list("ABCDEFGHIJ"), 2000), "full_future": y,
                         "good": np.clip(y * 0.8 + 0.1, 0, 1), "noise": rng.random(2000)})
    m = full_risk.metrics(y, pred["good"].to_numpy())
    assert m["auc"] == 1.0 and m["ece"] < 0.2
    d = full_risk.bootstrap_diff(pred, "good", "noise", "auc", reps=50)
    assert d["diff"] > 0.4 and d["ci95"][0] > 0.4
