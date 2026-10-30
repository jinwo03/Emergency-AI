from datetime import datetime

from contracts.schemas import HospitalDetail
from egen import store


def get_hospital(hpid: str) -> HospitalDetail | None:
    """병원 상세 정보 (수집된 최신 데이터). 없는 hpid 면 None."""
    with store.connect() as conn:
        row = store.hospital_detail(conn, hpid)
        if row is None or row["lat"] is None:
            return None
        notices = store.active_messages(conn, [hpid]).get(hpid, [])

    return HospitalDetail(
        hpid=row["hpid"],
        name=row["name"],
        address=row["address"],
        tel=row["tel"],
        er_tel=row["er_tel"],
        lat=row["lat"],
        lon=row["lon"],
        beds_available=row["hvec"],
        updated_at=datetime.fromisoformat(row["hvidate"]) if row["hvidate"] else None,
        notices=notices,
    )
