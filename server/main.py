"""앱 ↔ 서버 연결부 (규칙 ①). 알고리즘은 algorithm 패키지의 함수만 호출한다.

실행: uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
문서: http://localhost:8000/docs
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import algorithm
from contracts.schemas import (
    ErrorResponse,
    HospitalDetail,
    RecommendationRequest,
    RecommendationResponse,
    TriageRequest,
    TriageResponse,
    TriageStatus,
)

KST = timezone(timedelta(hours=9))
log = logging.getLogger("server")

app = FastAPI(title="Emergency-AI API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ERROR_RESPONSES = {400: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
                   422: {"model": ErrorResponse}, 500: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}}

# 되묻기 세션: session_id → 이전 발화 목록. 서버 재시작 시 초기화됨.
_sessions: dict[str, list[str]] = {}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(ApiError)
async def _api_error(_: Request, exc: ApiError):
    return _error(exc.status, exc.code, exc.message)


@app.exception_handler(algorithm.DataUnavailableError)
async def _data_unavailable(_: Request, exc: algorithm.DataUnavailableError):
    return _error(503, "DATA_UNAVAILABLE", str(exc))


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    field = ".".join(str(p) for p in first["loc"][1:])
    return _error(422, "VALIDATION_ERROR", f"{field}: {first['msg']}")


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, "HTTP_ERROR", str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("unhandled error")
    return _error(500, "INTERNAL_ERROR", "서버 내부 오류")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/triage", response_model=TriageResponse, responses=ERROR_RESPONSES)
def triage(req: TriageRequest):
    if req.session_id is None:
        session_id = uuid.uuid4().hex
        history: list[str] = []
    elif req.session_id in _sessions:
        session_id = req.session_id
        history = _sessions[session_id]
    else:
        raise ApiError(404, "SESSION_NOT_FOUND", "세션이 만료되었거나 존재하지 않습니다. session_id 없이 다시 시작하세요.")

    result = algorithm.analyze_symptoms(req.text, req.age, list(history))

    if result.status == TriageStatus.need_more_info:
        _sessions[session_id] = history + [req.text]
    else:
        _sessions.pop(session_id, None)

    return TriageResponse(**result.model_dump(), session_id=session_id)


@app.post("/v1/recommendations", response_model=RecommendationResponse, responses=ERROR_RESPONSES)
def recommendations(req: RecommendationRequest):
    result = algorithm.rank_hospitals(req.lat, req.lon, req.ktas, req.departments, req.conditions, req.limit)
    return RecommendationResponse(**result.model_dump(), generated_at=datetime.now(KST))


@app.get("/v1/hospitals/{hpid}", response_model=HospitalDetail, responses=ERROR_RESPONSES)
def hospital(hpid: str):
    detail = algorithm.get_hospital(hpid)
    if detail is None:
        raise ApiError(404, "HOSPITAL_NOT_FOUND", f"병원을 찾을 수 없습니다: {hpid}")
    return detail
