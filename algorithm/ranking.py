"""병원 순위 산출.

현재는 베이스라인 (c) '규칙 필터 + 거리순' 만 구현되어 있어 항상 mode=fallback 으로 응답한다.
TODO(알고리즘 담당): LightGBM 수용 확률 + 랭킹 모델을 붙이고, 신뢰도가 충분하면 mode=ai 로 응답.
"""
import math
from datetime import datetime

from algorithm.errors import DataUnavailableError
from contracts.schemas import RankedHospital, RecommendationMode, RecommendationResult
from egen import store

CANDIDATE_POOL = 30          # 가까운 순으로 이만큼 뽑은 뒤 규칙 정렬
ROAD_FACTOR = 1.3            # 직선거리 → 도로거리 보정
AVG_SPEED_KMH = 40           # 임시 ETA 가정. TODO: 지도 API(카카오/OSRM) ETA 로 교체
FALLBACK_WARNING = (
    "AI 예측 모델 준비 중입니다. 가용 병상이 있는 가까운 병원 순으로 보여드립니다. "
    "출발 전 병원에 전화로 수용 가능 여부를 확인하세요."
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def estimate_eta_min(distance_km: float) -> int:
    return max(1, round(distance_km * ROAD_FACTOR / AVG_SPEED_KMH * 60))


def rank_hospitals(
    lat: float,
    lon: float,
    ktas: int,
    departments: list[str],
    conditions: list[str],
    limit: int,
) -> RecommendationResult:
    """현재 위치와 환자 상태로 병원 순위를 산출한다.

    신뢰도가 낮으면 mode=fallback, accept_prob=None, warning 에 안내 문구를 채운다.
    hospitals 는 rank 오름차순, 최대 limit 개.
    """
    now = store.now_kst()
    with store.connect() as conn:
        states = store.current_hospital_states(conn)
        if not states:
            raise DataUnavailableError("수집된 병상 데이터가 없습니다. 수집기를 실행하세요.")

        nearest = sorted(states, key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))[:CANDIDATE_POOL]
        notices = store.active_messages(conn, [s["hpid"] for s in nearest])

    # 규칙: 가용 병상 > 0 인 곳 우선, 그 안에서 거리순
    def sort_key(s):
        has_beds = s["hvec"] is not None and s["hvec"] > 0
        return (not has_beds, haversine_km(lat, lon, s["lat"], s["lon"]))

    hospitals = []
    for rank, s in enumerate(sorted(nearest, key=sort_key)[:limit], start=1):
        dist = haversine_km(lat, lon, s["lat"], s["lon"])
        age_min = int((now - datetime.fromisoformat(s["hvidate"])).total_seconds() // 60)
        beds = s["hvec"]
        beds_text = "정보 없음" if beds is None else (f"{beds}개" if beds > 0 else "없음(과밀)")
        hospitals.append(RankedHospital(
            rank=rank,
            hpid=s["hpid"],
            name=s["name"],
            tel=s["er_tel"] or s["tel"],
            lat=s["lat"],
            lon=s["lon"],
            distance_km=round(dist, 2),
            eta_min=estimate_eta_min(dist),
            accept_prob=None,
            beds_available=beds,
            data_age_min=max(0, age_min),
            notices=notices.get(s["hpid"], []),
            reason=f"직선거리 {dist:.1f}km, 응급실 가용 병상 {beds_text} (거리순 기본 추천)",
        ))

    return RecommendationResult(mode=RecommendationMode.fallback, warning=FALLBACK_WARNING, hospitals=hospitals)
