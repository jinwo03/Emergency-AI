"""테스트는 실제 수집 DB 대신 임시 DB + 고정 샘플 데이터를 사용한다."""
import os
import tempfile
from datetime import timedelta
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["EGEN_DB_PATH"] = str(Path(_tmp) / "test.db")

import pytest  # noqa: E402

from egen import store  # noqa: E402


def _egen_dtm(dt) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


@pytest.fixture(autouse=True)
def sample_db():
    path = store.db_path()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)

    now = store.now_kst()
    seen = store.iso(now)
    with store.connect() as conn:
        store.save_hospitals(conn, [
            {"hpid": "A1100010", "dutyName": "삼성서울병원", "dutyAddr": "서울특별시 강남구 일원로 81",
             "dutyTel1": "02-3410-2114", "dutyTel3": "02-3410-2114", "wgs84Lat": "37.4885", "wgs84Lon": "127.0867"},
            {"hpid": "A1100015", "dutyName": "강남세브란스병원", "dutyTel1": "02-2019-3114",
             "wgs84Lat": "37.4926", "wgs84Lon": "127.0462"},
            {"hpid": "A1100043", "dutyName": "강동경희대학교병원", "dutyTel1": "02-440-7114",
             "wgs84Lat": "37.5534", "wgs84Lon": "127.1574"},
        ], seen)
        store.save_bed_snapshots(conn, [
            {"hpid": "A1100010", "hvidate": _egen_dtm(now - timedelta(minutes=3)), "hvec": "5"},
            {"hpid": "A1100015", "hvidate": _egen_dtm(now - timedelta(minutes=12)), "hvec": "-2"},  # 과밀
            {"hpid": "A1100043", "hvidate": _egen_dtm(now - timedelta(minutes=1)), "hvec": "2"},
        ], seen)
        store.save_messages(conn, [
            {"hpid": "A1100010", "symBlkMsgTyp": "중증", "symTypCod": "Y0042", "symTypCodMag": "대동맥응급(복부)",
             "trtPrtCodMag": "외과", "symBlkMsg": "의료진 부족",
             "symBlkSttDtm": _egen_dtm(now - timedelta(hours=1)), "symBlkEndDtm": _egen_dtm(now + timedelta(days=1))},
            {"hpid": "A1100010", "symBlkMsgTyp": "중증", "symTypCod": "Y0131", "symTypCodMag": "수족지접합",
             "trtPrtCodMag": "정형외과", "symBlkMsg": "의료진 부족",
             "symBlkSttDtm": _egen_dtm(now - timedelta(hours=1)), "symBlkEndDtm": _egen_dtm(now + timedelta(days=1))},
            {"hpid": "A1100010", "symBlkMsgTyp": "응급", "symTypCod": "Y000", "symTypCodMag": "응급실",
             "symBlkMsg": "종료된 메시지",
             "symBlkSttDtm": _egen_dtm(now - timedelta(days=2)), "symBlkEndDtm": _egen_dtm(now - timedelta(days=1))},
        ], seen)
        store.log_run(conn, seen, "messages", ok=True)
    yield
