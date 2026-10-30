"""앱 ↔ 서버 ↔ 알고리즘 간 데이터 계약 (규칙 ①, ② 공통).

규칙
- 필드 추가는 자유, 삭제/이름 변경/타입 변경은 팀 합의 후에만.
- 단위: 시간=분(min), 확률=0~1, 거리=km, 시각=ISO 8601 (+09:00).
- 값이 없으면 null. 빈 문자열("")로 대신하지 않는다.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------- 공통 ----------

class ErrorBody(BaseModel):
    code: str = Field(..., examples=["UPSTREAM_TIMEOUT"])
    message: str = Field(..., examples=["E-Gen API 응답 지연"])


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---------- 증상 분석 (POST /v1/triage) ----------

class TriageStatus(str, Enum):
    complete = "complete"              # 분석 완료 → 추천 호출 가능
    need_more_info = "need_more_info"  # follow_up_question 을 보호자에게 물어볼 것


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="STT 변환된 보호자 발화",
                      examples=["아버지가 갑자기 가슴이 아프다고 쓰러지셨어요"])
    age: int | None = Field(None, ge=0, le=120, description="환자 나이 (모르면 null)")
    session_id: str | None = Field(None, description="되묻기 이어가기용. 첫 요청은 null")


class TriageResult(BaseModel):
    """알고리즘 analyze_symptoms() 의 반환값."""
    status: TriageStatus
    ktas: int | None = Field(None, ge=1, le=5, description="추정 KTAS 중증도 (1=가장 위급)")
    departments: list[str] = Field(default_factory=list, examples=[["순환기내과"]])
    conditions: list[str] = Field(default_factory=list, description="의심 중증질환 코드", examples=[["AMI"]])
    follow_up_question: str | None = Field(None, examples=["지금 의식이 있으신가요?"])


class TriageResponse(TriageResult):
    session_id: str


# ---------- 병원 추천 (POST /v1/recommendations) ----------

class RecommendationMode(str, Enum):
    ai = "ai"              # AI 랭킹 결과
    fallback = "fallback"  # 신뢰도 저하 → 거리순/가용병상순 베이스라인


class RecommendationRequest(BaseModel):
    lat: float = Field(..., ge=33.0, le=39.0, examples=[37.4979])
    lon: float = Field(..., ge=124.0, le=132.0, examples=[127.0276])
    ktas: int = Field(..., ge=1, le=5)
    departments: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    limit: int = Field(5, ge=1, le=20)


class RankedHospital(BaseModel):
    rank: int = Field(..., ge=1)
    hpid: str = Field(..., examples=["A1100010"])
    name: str
    tel: str | None = None
    lat: float
    lon: float
    distance_km: float
    eta_min: int = Field(..., description="예상 도착 시간 (분)")
    accept_prob: float | None = Field(None, ge=0, le=1, description="도착 시점 수용 확률. fallback 모드면 null")
    beds_available: int | None = Field(None, description="응급실 일반 가용 병상 수 (E-Gen hvec). 음수=과밀")
    data_age_min: int | None = Field(None, description="병상 정보가 몇 분 전 갱신됐는지 (Staleness)")
    notices: list[str] = Field(default_factory=list, description="수용 곤란/주의 메시지")
    reason: str | None = Field(None, description="추천 사유 (SHAP → LLM 자연어)")


class RecommendationResult(BaseModel):
    """알고리즘 rank_hospitals() 의 반환값."""
    mode: RecommendationMode
    warning: str | None = Field(None, description="fallback 시 사용자에게 보여줄 경고")
    hospitals: list[RankedHospital]


class RecommendationResponse(RecommendationResult):
    generated_at: datetime


# ---------- 병원 상세 (GET /v1/hospitals/{hpid}) ----------

class HospitalDetail(BaseModel):
    hpid: str
    name: str
    address: str | None = None
    tel: str | None = None
    er_tel: str | None = None
    lat: float
    lon: float
    beds_available: int | None = Field(None, description="응급실 일반 가용 병상 수 (E-Gen hvec). 음수=과밀")
    updated_at: datetime | None = Field(None, description="E-Gen 병상 정보 갱신 시각 (hvidate)")
    notices: list[str] = Field(default_factory=list)
