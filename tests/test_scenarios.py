"""평가용 시나리오 파일 형식 검사 (진료과·질환 코드가 허용 목록 안에 있는지)."""
from algorithm.egen_codes import DEPARTMENTS, EGEN_CONDITIONS
import pytest

from eval.run_eval import HOLDOUT, SCENARIOS, STRESS, load_scenarios


@pytest.mark.parametrize("path,minimum", [(SCENARIOS, 200), (HOLDOUT, 50), (STRESS, 25)])
def test_scenarios_are_valid(path, minimum):
    rows = load_scenarios(path)
    assert len(rows) >= minimum
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["text"].strip()
        if r["ktas"] is None:          # 되묻기 기대 케이스
            continue
        assert 1 <= r["ktas"] <= 5, r["id"]
        assert r["departments"], r["id"]
        assert all(d in DEPARTMENTS for d in r["departments"]), r["id"]
        assert all(c in EGEN_CONDITIONS for c in r["conditions"]), r["id"]
        if r["cond_required"]:
            assert r["conditions"], r["id"]
