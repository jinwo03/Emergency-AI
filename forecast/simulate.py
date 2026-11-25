"""추천 방식 비교 시뮬레이션: 과거 시점에 가상 환자를 두고, 방식마다 고른 병원에 도착했을 때 실제로 받을 수 있었는지 본다.

실행 (forecast.full_risk 를 먼저 실행해 예측 파일이 있어야 한다):
    python -m forecast.simulate                    # 결과: data/models/simulation_report.json
    python -m forecast.simulate --patients 5000 --penalty 15 30 60

가정 (발표 때 함께 밝힌다)
- 환자 위치: 무작위 병원 주변 반경 RADIUS_KM 안의 임의 지점 (병원이 많은 곳 = 사람이 많은 곳으로 근사).
- 환자 시각: 예측 파일의 시험 기간(모델이 학습하지 않은 기간) 안의 5분 격자 시각.
- 환자 유형: PROFILES 중 하나를 같은 확률로 고른다 (실제 비율 아님).
- 이동 시간: 앱과 같은 직선거리 × 도로 보정 ÷ 평균 속도 (algorithm.ranking.estimate_eta_min).
- 정답(도착 시점): 실제 기록된 가용 병상(hvec ≤ 0 이면 만실)과, 그 시각에 떠 있던 메시지를 환자에게 맞춘 결과
  (불가·기존 환자만이면 공지상 불가). 둘 중 하나면 '도착 실패'.
- 실패하면 HANDOVER_MIN 분 뒤 그 병원에서 같은 방식으로 다음 병원을 고른다. 두 번째도 실패하면 SECOND_FAIL_MIN 분을 더한다.

방식
- nearest: 가장 가까운 병원
- nearest_beds_now: 지금 병상이 있는 가장 가까운 병원 (기존 서비스와 비슷)
- nearest_beds_msg: 위 + 지금 떠 있는 메시지로 이 환자를 못 받는 병원 제외
- risk_only: 이동 시간 + 벌점 × 도착 시 만실 확률 (메시지 안 씀)
- proposed: 이동 시간 + 벌점 × 실패 확률. 실패 확률 = 1 - (1 - 만실 확률)(1 - 메시지 위험)
  메시지 위험: 불가·기존 환자만 1, 어려움 0.5, 그 외 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm.message_match import Patient, status_for
from algorithm.ranking import estimate_eta_min
from egen import store

RADIUS_KM = 10
CANDIDATES = 20
MAX_ETA = 90
HANDOVER_MIN = 10
SECOND_FAIL_MIN = 30
STALE_SLOTS = 6            # 실제 병상 값이 비면 30분 이내 직전 값을 쓴다
BLOCKED = {"불가", "기존 환자만"}
MSG_RISK = {"불가": 1.0, "기존 환자만": 1.0, "어려움": 0.5}
MODEL = "lgbm_base"

PROFILES = {
    "성인 복통": Patient.of(["외과", "소화기내과"], [], ["성인"]),
    "뇌경색 의심": Patient.of(["신경과"], ["Y0020"], ["성인", "중증"]),
    "심근경색 의심": Patient.of(["순환기내과"], ["Y0010"], ["성인", "중증"]),
    "소아 발열": Patient.of(["소아청소년과"], [], ["소아", "경증"]),
    "성인 골절": Patient.of(["정형외과"], [], ["성인", "외상"]),
    "눈 외상": Patient.of(["안과"], [], ["성인", "외상"]),
    "임신 중 복통": Patient.of(["산부인과"], ["Y0111"], ["성인", "임산부"]),
}
POLICIES = ["nearest", "nearest_beds_now", "nearest_beds_msg", "risk_only", "proposed"]


def km(lat1, lon1, lat2, lon2):
    """벡터 haversine (km)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


class World:
    """병상(지금·실제), 예측 확률, 메시지를 시각·병원으로 찾는다."""

    def __init__(self, processed: Path, models: Path):
        grid = pd.read_parquet(processed / "beds_grid.parquet",
                               columns=["hpid", "t", "hvec", "is_stale", "collection_gap", "hvec_suspect"])
        hospitals = pd.read_parquet(processed / "hospitals_clean.parquet")
        hospitals = hospitals[~hospitals["exclude_candidate"]].reset_index(drop=True)
        grid = grid[grid["hpid"].isin(hospitals["hpid"])]
        ok = ~grid["is_stale"] & ~grid["collection_gap"] & (grid["hvec_suspect"] != True)
        hvec = grid[ok].pivot(index="t", columns="hpid", values="hvec").astype(float)
        self.hospitals = hospitals.set_index("hpid").loc[hvec.columns.intersection(hospitals["hpid"])]
        self.hvec = hvec[self.hospitals.index].ffill(limit=STALE_SLOTS)
        self.slots = self.hvec.index
        self.preds = {}
        for horizon in (30, 60):
            p = pd.read_parquet(models / f"full_risk_predictions_{horizon}min.parquet", columns=["t", "hpid", MODEL])
            self.preds[horizon] = dict(zip(zip(p["t"], p["hpid"]), p[MODEL]))
        test_t = pd.DatetimeIndex([t for t, _ in self.preds[30]])
        self.test_start, self.test_end = test_t.min(), test_t.max()
        self.messages = self._load_messages(processed)

    def _load_messages(self, processed: Path) -> dict[str, list[tuple]]:
        """병원 → [(게시 시작, 게시 끝, 줄 목록)]. 수집 결과에 처음 보인 때부터 마지막으로 보인 때(게시 종료 전)까지."""
        m = pd.read_parquet(processed / "message_targets.parquet")
        stop = m["last_seen"].where(m["last_seen"] <= m["end_dtm"], m["end_dtm"])
        m = m.assign(stop=stop)
        m = m[(m["stop"] >= self.test_start) & (m["first_seen"] <= self.test_end + pd.Timedelta(minutes=3 * MAX_ETA))]
        m = m.sort_values(["hpid", "series_id", "episode_no", "line_no"])
        cols = ["target", "departments", "disease_codes", "conditions", "time", "status", "caution"]
        out: dict[str, list[tuple]] = {}
        for (h, _, _), part in m.groupby(["hpid", "series_id", "episode_no"], sort=False):
            out.setdefault(h, []).append((part["first_seen"].iloc[0], part["stop"].iloc[0], part[cols].values.tolist()))
        return out

    def slot(self, at: pd.Timestamp) -> pd.Timestamp:
        return at.floor("5min")

    def beds(self, h: str, at: pd.Timestamp) -> float:
        s = self.slot(at)
        return float(self.hvec.at[s, h]) if s in self.slots else np.nan

    def msg_status(self, h: str, posted_at: pd.Timestamp, patient: Patient, at: pd.Timestamp) -> str:
        """posted_at 에 떠 있던 메시지로 본 시각 at 의 이 환자 상태. 가장 나쁜 메시지를 따른다."""
        worst = "가능"
        order = ["가능", "판단 불가", "미해석", "어려움", "기존 환자만", "불가"]
        naive = at.to_pydatetime().replace(tzinfo=None)
        for start, stop, lines in self.messages.get(h, []):
            if not start <= posted_at <= stop:
                continue
            status, _, _ = status_for(lines, patient, naive)
            if order.index(status) > order.index(worst):
                worst = status
        return worst

    def p_full(self, h: str, at: pd.Timestamp, eta: float) -> float:
        """도착 시 만실 확률. 0·30·60분 사이는 선형으로 잇는다 (0분 = 지금 만실 여부)."""
        s = self.slot(at)
        now = self.beds(h, at)
        p0 = 1.0 if (not np.isnan(now) and now <= 0) else 0.0
        fallback = p0 if not np.isnan(now) else 0.5
        p30 = self.preds[30].get((s, h), fallback)
        p60 = self.preds[60].get((s, h), p30)
        if eta <= 30:
            return p0 + (p30 - p0) * eta / 30
        return p30 + (p60 - p30) * min(eta - 30, 30) / 30


def candidates(world: World, lat: float, lon: float, exclude: set) -> pd.DataFrame:
    h = world.hospitals
    d = km(lat, lon, h["lat"].to_numpy(), h["lon"].to_numpy())
    df = pd.DataFrame({"hpid": h.index, "km": d, "lat": h["lat"].to_numpy(), "lon": h["lon"].to_numpy()})
    df = df[~df["hpid"].isin(exclude)].nsmallest(CANDIDATES, "km")
    df["eta"] = [estimate_eta_min(x) for x in df["km"]]
    return df[df["eta"] <= MAX_ETA]


def choose(world: World, policy: str, patient: Patient, lat: float, lon: float, at: pd.Timestamp,
           exclude: set, penalty: float) -> tuple[str, int] | None:
    c = candidates(world, lat, lon, exclude)
    if c.empty:
        return None
    if policy == "nearest":
        r = c.iloc[0]
        return r["hpid"], int(r["eta"])
    if policy in ("nearest_beds_now", "nearest_beds_msg"):
        for r in c.itertuples():
            beds = world.beds(r.hpid, at)
            if np.isnan(beds) or beds <= 0:
                continue
            arrive = at + pd.Timedelta(minutes=int(r.eta))
            if policy == "nearest_beds_msg" and world.msg_status(r.hpid, at, patient, arrive) in BLOCKED:
                continue
            return r.hpid, int(r.eta)
        r = c.iloc[0]
        return r.hpid, int(r.eta)
    best, best_cost = None, np.inf
    for r in c.itertuples():
        p = world.p_full(r.hpid, at, r.eta)
        if policy == "proposed":
            arrive = at + pd.Timedelta(minutes=int(r.eta))
            q = MSG_RISK.get(world.msg_status(r.hpid, at, patient, arrive), 0.0)
            p = 1 - (1 - p) * (1 - q)
        cost = r.eta + penalty * p
        if cost < best_cost:
            best, best_cost = (r.hpid, int(r.eta)), cost
    return best


def outcome(world: World, h: str, patient: Patient, arrive: pd.Timestamp) -> dict:
    beds = world.beds(h, arrive)
    msg = world.msg_status(h, arrive, patient, arrive)
    full = bool(not np.isnan(beds) and beds <= 0)
    blocked = msg in BLOCKED
    return {"bed_full": full, "msg_blocked": blocked, "difficult": msg == "어려움",
            "unknown_beds": bool(np.isnan(beds)), "fail": full or blocked}


def run_patient(world: World, policy: str, patient: Patient, lat: float, lon: float, at: pd.Timestamp,
                penalty: float) -> dict | None:
    first = choose(world, policy, patient, lat, lon, at, set(), penalty)
    if first is None:
        return None
    h1, eta1 = first
    arrive1 = at + pd.Timedelta(minutes=eta1)
    o1 = outcome(world, h1, patient, arrive1)
    total, fails = eta1, int(o1["fail"])
    if o1["fail"]:
        h = world.hospitals.loc[h1]
        retry_at = arrive1 + pd.Timedelta(minutes=HANDOVER_MIN)
        second = choose(world, policy, patient, h["lat"], h["lon"], retry_at, {h1}, penalty)
        total += HANDOVER_MIN
        if second is None:
            total += SECOND_FAIL_MIN
            fails += 1
        else:
            h2, eta2 = second
            o2 = outcome(world, h2, patient, retry_at + pd.Timedelta(minutes=eta2))
            total += eta2 + (SECOND_FAIL_MIN if o2["fail"] else 0)
            fails += int(o2["fail"])
    return {"hospital": h1, "eta": eta1, "beds_now": world.beds(h1, at),
            **{f"first_{k}": v for k, v in o1.items()}, "total_min": total, "fails": fails}


def sample_patients(world: World, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = world.hospitals.iloc[rng.integers(0, len(world.hospitals), n)]
    r = RADIUS_KM * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2 * np.pi
    lat = base["lat"].to_numpy() + r * np.cos(theta) / 111.0
    lon = base["lon"].to_numpy() + r * np.sin(theta) / (111.0 * np.cos(np.radians(base["lat"].to_numpy())))
    latest = world.test_end - pd.Timedelta(minutes=2 * MAX_ETA + HANDOVER_MIN)
    slots = world.slots[(world.slots >= world.test_start) & (world.slots <= latest)]
    return pd.DataFrame({"lat": lat, "lon": lon, "t": slots[rng.integers(0, len(slots), n)],
                         "profile": rng.choice(list(PROFILES), n)})


def summarize(df: pd.DataFrame) -> dict:
    return {
        "patients": len(df),
        "first_fail_rate": round(float(df["first_fail"].mean()), 4),
        "first_bed_full_rate": round(float(df["first_bed_full"].mean()), 4),
        "first_msg_blocked_rate": round(float(df["first_msg_blocked"].mean()), 4),
        "first_difficult_rate": round(float(df["first_difficult"].mean()), 4),
        "two_fails_rate": round(float((df["fails"] >= 2).mean()), 4),
        "eta_mean": round(float(df["eta"].mean()), 2),
        "total_min_mean": round(float(df["total_min"].mean()), 2),
        "total_min_p90": round(float(df["total_min"].quantile(0.9)), 1),
        "unknown_beds_rate": round(float(df["first_unknown_beds"].mean()), 4),
    }


def bootstrap_ci(a: pd.Series, b: pd.Series, reps: int = 1000, seed: int = 0) -> list[float]:
    """같은 환자 짝 차이(a - b)의 평균 95% 구간."""
    d = (a - b).to_numpy(float)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
    return [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])]


def simulate(world: World, patients: pd.DataFrame, penalty: float) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(patients.itertuples()):
        patient = PROFILES[p.profile]
        for policy in POLICIES:
            r = run_patient(world, policy, patient, p.lat, p.lon, p.t, penalty)
            if r is not None:
                rows.append({"patient": i, "profile": p.profile, "policy": policy, **r})
    return pd.DataFrame(rows)


def report(results: pd.DataFrame) -> dict:
    complete = results.groupby("patient")["policy"].transform("nunique") == len(POLICIES)
    results = results[complete]
    out = {"overall": {pol: summarize(g) for pol, g in results.groupby("policy")}}
    # 기존 방식이 고른 병원이 지금 1~3자리뿐인 환자 (병상 위험 예측이 의미 있는 상황)
    base = results[results["policy"] == "nearest_beds_now"].set_index("patient")["beds_now"]
    near = base[base.between(1, 3)].index
    out["near_full_cases"] = {pol: summarize(g) for pol, g in results[results["patient"].isin(near)].groupby("policy")}
    long_eta = results.loc[(results["policy"] == "nearest_beds_now") & (results["eta"] >= 20), "patient"]
    out["eta_20min_plus_cases"] = {pol: summarize(g)
                                   for pol, g in results[results["patient"].isin(long_eta)].groupby("policy")}
    out["by_profile"] = {prof: {pol: summarize(g) for pol, g in part.groupby("policy")}
                         for prof, part in results.groupby("profile")}
    wide = {k: results.pivot(index="patient", columns="policy", values=k).astype(float)
            for k in ("first_fail", "total_min")}
    out["paired_vs_nearest_beds_now"] = {
        pol: {k: {"diff": round(float((wide[k][pol] - wide[k]["nearest_beds_now"]).mean()), 4),
                  "ci95": bootstrap_ci(wide[k][pol], wide[k]["nearest_beds_now"])} for k in wide}
        for pol in POLICIES if pol != "nearest_beds_now"}
    return out


def main() -> None:
    data = store.DEFAULT_DB.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=data / "processed")
    parser.add_argument("--models", type=Path, default=data / "models")
    parser.add_argument("--patients", type=int, default=3000)
    parser.add_argument("--penalty", type=float, nargs="+", default=[30.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    world = World(args.processed, args.models)
    patients = sample_patients(world, args.patients, args.seed)
    out = {"assumptions": {"patients": args.patients, "radius_km": RADIUS_KM, "handover_min": HANDOVER_MIN,
                           "second_fail_min": SECOND_FAIL_MIN, "profiles": list(PROFILES), "model": MODEL,
                           "period": [str(world.test_start), str(world.test_end)]}}
    for penalty in args.penalty:
        results = simulate(world, patients, penalty)
        results.to_parquet(args.models / f"simulation_results_p{int(penalty)}.parquet", index=False)
        out[f"penalty_{int(penalty)}"] = report(results)
        print(f"벌점 {penalty:g}분 완료", flush=True)
    path = args.models / "simulation_report.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
