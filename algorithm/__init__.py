"""알고리즘 담당 영역. 서버는 아래 세 함수만 호출한다 (규칙 ②).

시그니처(이름·인자·반환 타입)는 팀 합의 없이 바꾸지 않는다.
내부 구현, 하위 모듈 추가는 자유.
데이터가 없어 답할 수 없으면 DataUnavailableError 를 던진다.
"""
from algorithm.errors import DataUnavailableError
from algorithm.hospitals import get_hospital
from algorithm.ranking import rank_hospitals
from algorithm.triage import analyze_symptoms

__all__ = ["analyze_symptoms", "rank_hospitals", "get_hospital", "DataUnavailableError"]
