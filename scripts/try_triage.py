"""실제 Ollama로 증상 분석을 돌려보는 스크립트.

사용: .venv/Scripts/python.exe -m scripts.try_triage
모델 바꾸기: $env:TRIAGE_MODEL="exaone3.5:7.8b" 후 실행
"""
import os
import time

from algorithm import triage

os.environ["TRIAGE_BACKEND"] = "llm"

SAMPLES = [
    ("아버지가 갑자기 가슴이 아프다고 쓰러지셨어요", 68),
    ("엄마가 갑자기 말이 어눌해지고 오른쪽 팔에 힘이 없어요", 70),
    ("아이가 동전을 삼킨 것 같아요 계속 침을 흘려요", 3),
    ("남편이 배가 너무 아프대요 오른쪽 아랫배요 열도 나요", 35),
    ("아내가 임신 30주인데 피가 나고 배가 아파요", 32),
    ("할아버지가 검은 변을 보시고 어지럽다고 하세요", 82),
]

if __name__ == "__main__":
    print("모델:", os.environ.get("TRIAGE_MODEL", "qwen2.5:7b"))
    for text, age in SAMPLES:
        start = time.time()
        r = triage.analyze_symptoms(text, age, [])
        sec = time.time() - start
        print(f"\n[{sec:.1f}초] {text}")
        print(f"  status={r.status.value} ktas={r.ktas} 진료과={r.departments} 질환={r.conditions}")
        if r.follow_up_question:
            print(f"  되묻기: {r.follow_up_question}")
