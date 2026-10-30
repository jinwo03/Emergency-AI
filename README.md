# Emergency-AI
Intelligent Emergency Department Recommendation System Based on Real-Time Medical Data and ETA-Aware Acceptance Probability Prediction

ETA 기반 응급실 수용 확률 예측 · 추천 시스템.

```
[앱]  ──규칙①: HTTP/JSON──▶  [server/ FastAPI]  ──규칙②: 파이썬 함수──▶  [algorithm/]
                                                                         │
                                                                    [E-Gen API]
```

## 폴더 구조

| 경로 | 담당 | 설명 |
|---|---|---|
| `contracts/schemas.py` | **공동 (합의 필수)** | 앱·서버·알고리즘이 주고받는 데이터 형식 |
| `algorithm/` | 알고리즘 담당 | `analyze_symptoms`, `rank_hospitals`, `get_hospital` 구현 |
| `server/main.py` | 서버(연결) 담당 | 엔드포인트 ↔ 알고리즘 함수 연결, 세션, 에러 처리 |
| `egen/` | 데이터 담당 | E-Gen API 클라이언트(`client.py`), SQLite 저장소(`store.py`), 5분 주기 수집기(`collector.py`) |
| `data/` | (git 제외) | `egen.db` 수집 DB, `collector.log` |
| `tests/test_contract.py` | 공동 | 계약 테스트. 항상 통과해야 함 |
| `scripts/` | - | 일회성 스크립트 (E-Gen API 테스트 등) |

## 규칙 ① 앱 ↔ 서버 (HTTP)

서버 실행 후 **http://<서버IP>:8000/docs** 에서 전체 명세 확인 및 직접 호출 가능.

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/v1/triage` | 증상 텍스트 → 중증도/진료과, 또는 되묻기 질문 |
| POST | `/v1/recommendations` | 위치 + 중증도 → 병원 순위 |
| GET | `/v1/hospitals/{hpid}` | 병원 상세 |
| GET | `/health` | 서버 살아있는지 확인 |

**앱 흐름**
1. `/v1/triage` 에 `{ "text": ..., "session_id": null }`
2. `status == "need_more_info"` 면 `follow_up_question` 을 보여주고, 답변을 **같은 `session_id`** 로 다시 `/v1/triage`
3. `status == "complete"` 면 받은 `ktas`, `departments`, `conditions` + 현재 위치로 `/v1/recommendations`
4. `mode == "fallback"` 이면 `warning` 을 눈에 띄게 표시 (AI 순위가 아니라 거리/병상순)

**공통 규칙**
- 단위: 시간=분, 확률=0~1, 거리=km, 시각=ISO 8601 `+09:00`
- 값이 없으면 `null`
- 에러는 항상 `{ "error": { "code": "...", "message": "..." } }` — 앱은 `code` 로 분기
- 필드 **추가**는 자유, **삭제·이름 변경·타입 변경은 팀 합의 후** `contracts/schemas.py` 수정

## 규칙 ② 서버 ↔ 알고리즘 (파이썬 함수)

`algorithm/__init__.py` 의 세 함수 시그니처는 합의 없이 바꾸지 않는다. 내부 구현과 하위 모듈은 자유.

```python
analyze_symptoms(text: str, age: int | None, history: list[str]) -> TriageResult
rank_hospitals(lat, lon, ktas, departments, conditions, limit) -> RecommendationResult
get_hospital(hpid: str) -> HospitalDetail | None
```

현재 상태:
- `get_hospital`: 수집 DB 실데이터
- `rank_hospitals`: 수집 DB 실데이터 + 베이스라인(가용 병상 있는 곳 우선 → 거리순), 항상 `mode=fallback`
- `analyze_symptoms`: 목(mock) 규칙

알고리즘 담당은 `TODO` 부분을 실제 로직으로 교체하고 `pytest` 통과를 확인한 뒤 넘긴다. 서버 코드는 수정할 필요 없음.
데이터가 없어 답할 수 없으면 `DataUnavailableError` 를 던지면 서버가 503 `DATA_UNAVAILABLE` 로 응답한다.
테스트는 `tests/conftest.py` 의 임시 DB 를 쓰므로 실제 수집 데이터에 영향 없음.

## 실행

```bash
# 최초 1회
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
cp .env.example .env   # EGEN_SERVICE_KEY 입력 (서버 PC에만)

# 테스트
.venv/Scripts/python.exe -m pytest

# 수집기 (서버 PC 에서 상시 실행. 5분마다 전국 병상/메시지, 24시간마다 기관 목록)
.venv/Scripts/python.exe -m egen.collector          # --once 로 1회만 실행 가능

# 서버 (같은 와이파이의 폰에서 접속하려면 --host 0.0.0.0)
.venv/Scripts/python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

앱에서 접속: 같은 와이파이면 `http://<서버PC 내부IP>:8000` (`ipconfig` 의 IPv4 주소, Windows 방화벽 8000 포트 허용 필요). 외부에서는 Cloudflare Tunnel / ngrok 주소 사용.
