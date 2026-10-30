class DataUnavailableError(Exception):
    """추천에 필요한 데이터가 없음 (수집기 미실행 등). 서버는 503 으로 응답한다."""
