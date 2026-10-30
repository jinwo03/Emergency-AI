from contracts.schemas import TriageResult, TriageStatus


def analyze_symptoms(text: str, age: int | None, history: list[str]) -> TriageResult:
    """보호자 발화에서 중증도/필요 진료과를 추출한다.

    Args:
        text: 이번 발화 (STT 결과)
        age: 환자 나이, 모르면 None
        history: 같은 세션의 이전 발화들 (오래된 순). 첫 요청이면 빈 리스트.

    Returns:
        정보가 충분하면 status=complete + ktas/departments,
        부족하면 status=need_more_info + follow_up_question.

    TODO(알고리즘 담당): Local LLM 구조화 출력으로 교체. 아래는 목(mock) 구현.
    """
    if len(text) < 10 and not history:
        return TriageResult(
            status=TriageStatus.need_more_info,
            follow_up_question="어떤 증상이 있으신지 조금 더 자세히 말씀해 주세요.",
        )

    if "가슴" in text or "흉통" in text:
        return TriageResult(status=TriageStatus.complete, ktas=2,
                            departments=["순환기내과"], conditions=["AMI"])

    return TriageResult(status=TriageStatus.complete, ktas=3, departments=["응급의학과"])
