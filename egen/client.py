"""E-Gen(국립중앙의료원 ErmctInfoInqireService) API 클라이언트."""
import os
from pathlib import Path
from urllib.parse import unquote
from xml.etree.ElementTree import ParseError

import defusedxml.ElementTree as ET
import requests
from dotenv import load_dotenv

BASE_URL = "https://apis.data.go.kr/B552657/ErmctInfoInqireService"
PAGE_SIZE = 1000

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class EgenError(Exception):
    pass


def _service_key() -> str:
    key = os.getenv("EGEN_SERVICE_KEY", "").strip()
    if not key:
        raise EgenError("EGEN_SERVICE_KEY 가 .env 에 없습니다.")
    return unquote(key)


def _fetch_all(operation: str, params: dict | None = None, timeout: float = 60) -> list[dict]:
    """페이지를 끝까지 돌며 item 을 dict 리스트로 반환. 값의 앞뒤 공백은 제거."""
    items: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            f"{BASE_URL}/{operation}",
            params={"serviceKey": _service_key(), "pageNo": page, "numOfRows": PAGE_SIZE, **(params or {})},
            timeout=timeout,
        )
        try:
            root = ET.fromstring(resp.content)
        except ParseError as e:
            raise EgenError(f"{operation}: XML 파싱 실패 (HTTP {resp.status_code})") from e

        auth_err = root.findtext(".//returnAuthMsg")
        if auth_err:
            raise EgenError(f"{operation}: {auth_err} (reasonCode={root.findtext('.//returnReasonCode')})")
        code = root.findtext(".//resultCode")
        if code != "00":
            raise EgenError(f"{operation}: resultCode={code} {root.findtext('.//resultMsg')}")

        page_items = [{c.tag: (c.text or "").strip() for c in it} for it in root.findall(".//item")]
        items.extend(page_items)
        total = int(root.findtext(".//totalCount") or 0)
        if not page_items or len(items) >= total:
            return items
        page += 1


def fetch_realtime_beds() -> list[dict]:
    """전국 응급실 실시간 가용병상."""
    return _fetch_all("getEmrrmRltmUsefulSckbdInfoInqire")


def fetch_messages() -> list[dict]:
    """전국 응급실 및 중증질환 메시지 (수용 곤란 고지)."""
    return _fetch_all("getEmrrmSrsillDissMsgInqire")


def fetch_hospitals() -> list[dict]:
    """전국 응급의료기관 목록 (좌표 포함)."""
    return _fetch_all("getEgytListInfoInqire")
