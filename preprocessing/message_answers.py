"""응급 메시지 문구의 해석 결과를 한곳에 모은다.

순서: 정답 시트의 답(source=label) → 임시 시험용 답(source=provisional, 사람 입력 전까지 Claude 가 단 답)
→ 분류 모델(source=classifier, classify_messages --apply 결과). 앞의 답이 있으면 뒤의 것은 쓰지 않는다.
셋 다 없으면 message_status 가 문구 규칙(text_rule)을 쓴다.

LLM 해석(parse_messages)은 정답 초안을 만드는 데만 쓰고 여기서는 읽지 않는다 (LLM 은 추천 결정에 쓰지 않는다).
결과는 정규화 문구 → {"source", "reason", "lines": [[대상, 진료과, 질환, 조건, 시간, 상태, 주의], ...], ...}.
"""
from pathlib import Path

from preprocessing import label_set, message_status

CLASSIFIER_FILE = "message_cls.jsonl"   # classify_messages.APPLY_FILE (torch 없이 읽으려고 이름만 같게 둔다)


def load(labels_dir: Path, provisional: bool = True) -> dict[str, dict]:
    labels, _ = label_set.load_label_answers(labels_dir)
    classified = message_status.load_parses(labels_dir / CLASSIFIER_FILE)
    answers = {k: {**v, "source": "classifier"} for k, v in classified.items()}
    if provisional:
        answers.update({k: {"source": "provisional", **v}
                        for k, v in label_set.load_provisional_answers(labels_dir).items()})
    answers.update({k: {"source": "label", **v} for k, v in labels.items()})
    return answers
