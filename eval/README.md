# Triage 평가

| 파일 | 용도 |
|---|---|
| `scenarios.csv` | 개발용 201건. 오류 분석과 규칙 튜닝에 사용 |
| `scenarios_holdout.csv` | 검증용 61건. 규칙 튜닝에 쓰지 않고 성능 보고에만 사용 |
| `v1_run/cases_qwen_v1.csv` | v1(Qwen 2.5 7B) 개발용 세트 평가 기록. `--replay`로 재사용 |
| `run_eval.py` | 평가 실행 |
| `compare.py` | 결과 비교표 출력 |

```powershell
.venv/Scripts/python.exe -m eval.run_eval --backend llm --set holdout
.venv/Scripts/python.exe -m eval.run_eval --backend llm --set holdout --no-safety   # 규칙 보정 끔 (LLM 단독)
.venv/Scripts/python.exe -m eval.run_eval --replay eval/v1_run/cases_qwen_v1.csv    # Ollama 없이 후처리만 비교
.venv/Scripts/python.exe -m eval.compare
```

주의: 개발용 세트 점수는 규칙을 그 세트의 오답을 보고 만들었기 때문에 실제보다 높게 나온다.
성능 보고는 검증용 세트 기준으로 한다.
