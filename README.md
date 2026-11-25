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
| `contracts/vocab.py` | **공동 (합의 필수)** | 진료과 이름·중증질환 Y코드·환자 조건 고정 목록 (추천 요청과 정답 시트가 함께 씀) |
| `algorithm/` | 알고리즘 담당 | `analyze_symptoms`, `rank_hospitals`, `get_hospital` 구현. 메시지 줄 × 환자 × 시각 → 상태 `message_match.py` |
| `server/main.py` | 서버(연결) 담당 | 엔드포인트 ↔ 알고리즘 함수 연결, 세션, 에러 처리 |
| `egen/` | 데이터 담당 | E-Gen API 클라이언트(`client.py`), SQLite 저장소(`store.py`), 5분 주기 수집기(`collector.py`) |
| `forecast/` | 데이터 담당 | 도착 시점 만실 위험 예측 실험 `full_risk.py`, 추천 방식 비교 시뮬레이션 `simulate.py` (전처리 결과를 읽음) |
| `preprocessing/` | 데이터 담당 | 수집 DB 를 읽기 전용으로 읽어 학습용 데이터 생성 (정리 `clean.py`, 5분 격자 `grid.py`, 메시지 줄 연결 `message_status.py`, 해석 결과 모음(정답 시트 → 분류 모델) `message_answers.py`, 메시지 분류 모델 `classify_messages.py`·시간 규칙 `message_time.py`, 정답 초안용 LLM 해석 `parse_messages.py`, 문구 정답 세트 `label_set.py`) |
| `data/` | (git 제외) | `egen.db` 수집 DB, `collector.log`, `processed/` 전처리 결과, `labels/` 정답 시트·분류 결과, `models/` 예측·분류 모델 |
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
3. `status == "complete"` 면 받은 `ktas`, `departments`, `conditions`, `patient_conditions` + 환자 나이 `age` + 현재 위치로 `/v1/recommendations`
4. `mode == "fallback"` 이면 `warning` 을 눈에 띄게 표시 (AI 순위가 아니라 거리/병상순)

**공통 규칙**
- 단위: 시간=분, 확률=0~1, 거리=km, 시각=ISO 8601 `+09:00`
- 값이 없으면 `null`
- 에러는 항상 `{ "error": { "code": "...", "message": "..." } }` — 앱은 `code` 로 분기
- 필드 **추가**는 자유, **삭제·이름 변경·타입 변경은 팀 합의 후** `contracts/schemas.py` 수정
- `departments` 는 `contracts/vocab.py` 의 진료과 이름, `conditions` 는 Y코드 (예: `Y0010` 심근경색 재관류중재술), `patient_conditions` 는 환자 조건 (예: `소아`, `중증`, `중환자실 필요`. 목록에 없는 값은 422)

## 규칙 ② 서버 ↔ 알고리즘 (파이썬 함수)

`algorithm/__init__.py` 의 세 함수 시그니처는 합의 없이 바꾸지 않는다. 내부 구현과 하위 모듈은 자유.

```python
analyze_symptoms(text: str, age: int | None, history: list[str]) -> TriageResult
rank_hospitals(lat, lon, ktas, departments, conditions, limit, age=None, patient_conditions=None) -> RecommendationResult
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

# 전처리 (원본 DB 는 읽기만 함. 수집기 실행 중에도 가능. 결과: data/processed/)
.venv/Scripts/python.exe -m preprocessing.run

# 응급 메시지 해석: 추천에는 분류 모델(KLUE-RoBERTa)만 쓴다. LLM 은 결정에 쓰지 않는다 (docs 원칙)
# 전처리는 정답 시트 답 → 임시 시험용 답 → 분류 모델 결과 → 문구 규칙 순으로 쓴다
.venv/Scripts/python.exe -m preprocessing.classify_messages --backend roberta --save   # 학습 + 저장 (data/models/message_classifier/, GPU 권장)
.venv/Scripts/python.exe -m preprocessing.classify_messages --apply   # 정답 시트에 없는 문구 분류 → data/labels/message_cls.jsonl. 끝나면 전처리를 다시 실행
# 정답 초안용 LLM 해석 (ANTHROPIC_API_KEY 필요, 비용 발생. 추천에는 쓰지 않음, 사람이 검수할 초안·비교 실험용)
.venv/Scripts/python.exe -m preprocessing.parse_messages --dry-run   # API 호출 없이 대상 수만 확인
.venv/Scripts/python.exe -m preprocessing.parse_messages
.venv/Scripts/python.exe -m preprocessing.parse_messages --retry-invalid   # 형식 오류였던 문구도 다시 제출
.venv/Scripts/python.exe -m preprocessing.parse_messages --export-review   # 검수용 CSV
.venv/Scripts/python.exe -m forecast.full_risk   # 도착 시점 만실 위험 예측 실험 (30·60분). 결과: data/models/full_risk_report.json
.venv/Scripts/python.exe -m forecast.simulate --patients 3000 --penalty 15 30 60   # 추천 방식 비교 (full_risk 먼저). 결과: data/models/simulation_report.json
.venv/Scripts/python.exe -m preprocessing.score_parses --details scores.csv   # 정답 시트와 비교한 점수 (학습용·학습용 검수분·시험용·시험용 중복 제외)
.venv/Scripts/python.exe -m preprocessing.score_parses --provisional   # 사람이 아직 안 채운 시험용은 임시 답(Claude 작성, data/labels/claude_provisional_test_answers_do_not_open.jsonl)으로 채점. 전처리도 사람 입력 전까지 이 임시 답을 씀
.venv/Scripts/python.exe -m preprocessing.parse_local   # 로컬 Qwen(Ollama, 비용 없음)으로 시험용 문구 해석 → data/labels/local_parse_<모델>.jsonl. 채점: score_parses --provisional --pred <그 파일>
.venv/Scripts/python.exe -m preprocessing.classify_messages --backend tfidf   # 분류기 방식(A): rule / tfidf / roberta(KLUE-RoBERTa, GPU) / oracle(한 줄 방식 상한) → data/labels/cls_parse_<방식>.jsonl

# 응급 메시지 문구 정답 세트 (data/labels/). 현재 시트의 학습용 정답·시험용 입력을 이어받고, 덮어쓰기 전에 .bak.xlsx 로 남김
.venv/Scripts/python.exe -m preprocessing.label_set --dump todo.jsonl          # 정답이 필요한 문구 목록
.venv/Scripts/python.exe -m preprocessing.label_set --answers answers.jsonl    # 문구 목록·정답 시트·숨긴 시험용 답 생성
.venv/Scripts/python.exe -m preprocessing.label_set --fresh --answers answers.jsonl  # 현재 시트를 이어받지 않고 새로 생성
.venv/Scripts/python.exe -m preprocessing.label_set --refresh   # DB 없이 시트를 지금 형식으로 다시 쓰기 (답·검수 O 유지, 게시 시간 갱신·정렬)
.venv/Scripts/python.exe -m preprocessing.label_set --audit     # 검수 진행 상황 + 무작위 표본 기준 AI 정답 오류율(95% 구간)

# 서버 (같은 와이파이의 폰에서 접속하려면 --host 0.0.0.0)
.venv/Scripts/python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

앱에서 접속: 같은 와이파이면 `http://<서버PC 내부IP>:8000` (`ipconfig` 의 IPv4 주소, Windows 방화벽 8000 포트 허용 필요). 외부에서는 Cloudflare Tunnel / ngrok 주소 사용.
