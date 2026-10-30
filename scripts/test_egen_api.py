"""E-Gen(국립중앙의료원 전국 응급의료기관 정보 조회 서비스) API 연결 테스트.

사용법:
    1) medical/.env 에 EGEN_SERVICE_KEY=<공공데이터포털 인증키> 작성
    2) python scripts/test_egen_api.py [시도] [시군구]
       예) python scripts/test_egen_api.py 서울특별시 강남구
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import requests
from dotenv import load_dotenv

BASE_URL = "https://apis.data.go.kr/B552657/ErmctInfoInqireService"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.stdout.reconfigure(encoding="utf-8")


def get_service_key() -> str:
    key = os.getenv("EGEN_SERVICE_KEY", "").strip()
    if not key:
        sys.exit("EGEN_SERVICE_KEY 가 .env 에 없습니다.")
    # 포털의 Encoding 키를 넣어도 동작하도록 디코딩 (requests 가 다시 인코딩함)
    return unquote(key)


def call(operation: str, params: dict, key: str) -> None:
    print(f"\n=== {operation} ===")
    try:
        resp = requests.get(
            f"{BASE_URL}/{operation}",
            params={"serviceKey": key, "pageNo": 1, "numOfRows": 3, **params},
            timeout=15,
        )
    except requests.RequestException as e:
        print(f"[FAIL] 요청 오류: {e}")
        return

    print(f"HTTP {resp.status_code}")
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        print(f"[FAIL] XML 아님: {resp.text[:300]}")
        return

    # 인증 실패 등은 OpenAPI_ServiceResponse 형태로 옴
    err = root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg")
    if err:
        print(f"[FAIL] {err} (reasonCode={root.findtext('.//returnReasonCode')})")
        return

    code = root.findtext(".//resultCode")
    msg = root.findtext(".//resultMsg")
    total = root.findtext(".//totalCount")
    items = root.findall(".//item")
    status = "OK" if code in ("00", "0000") else "FAIL"
    print(f"[{status}] resultCode={code} resultMsg={msg} totalCount={total}")
    for item in items[:2]:
        fields = {c.tag: (c.text or "") for c in item}
        preview = ", ".join(f"{k}={v}" for k, v in list(fields.items())[:8])
        print(f"  - {preview}")


def main() -> None:
    stage1 = sys.argv[1] if len(sys.argv) > 1 else "서울특별시"
    stage2 = sys.argv[2] if len(sys.argv) > 2 else "강남구"
    key = get_service_key()

    call("getEmrrmRltmUsefulSckbdInfoInqire", {"STAGE1": stage1, "STAGE2": stage2}, key)  # 실시간 가용병상
    call("getSrsillDissAceptncPosblInfoInqire", {"STAGE1": stage1, "STAGE2": stage2}, key)  # 중증질환 수용가능
    call("getEmrrmSrsillDissMsgInqire", {"Q0": stage1, "Q1": stage2}, key)  # 응급실/중증질환 메시지
    call("getEgytListInfoInqire", {"Q0": stage1, "Q1": stage2}, key)  # 기관 목록
    call("getEgytLcinfoInqire", {"WGS84_LON": "127.0276", "WGS84_LAT": "37.4979"}, key)  # 위치 기반 (강남역)


if __name__ == "__main__":
    main()
