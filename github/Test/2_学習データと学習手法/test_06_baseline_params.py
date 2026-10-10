"""06: ベースラインの設定（research/src/baseline_params.json）が、決めた形で、参照している重みと戦略の引数に合っていること。

json は W001/REP001 の成果物（以後の REP が、この 1 ファイルから既存 8 手法の設定を引く）。
重みのファイルが無い端末（50MB を超える BCQ の重みは git に入れていない）では、ファイルが要る確認だけスキップする。
"""
import inspect
import json

import pytest

import run_experiment as rx
import train_model as tm
from conftest import REPO_ROOT

PATH = REPO_ROOT / "research" / "src" / "baseline_params.json"
P = json.loads(PATH.read_text(encoding="utf-8")) if PATH.is_file() else None
RULE = ["PID", "ABid", "OnlineLP"]
LEARNED = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]
pytestmark = pytest.mark.skipif(P is None, reason="research/src/baseline_params.json が無い")


def test_top_level_shape():
    assert P["version"] == 1 and P["decided_in"] == "REP001"
    assert len(P["commit"]) == 40 and all(c in "0123456789abcdef" for c in P["commit"])
    assert [k for k in P["rule_based"] if k != "note"] == RULE
    assert list(P["learned"]) == LEARNED


def test_selection_days_do_not_overlap_evaluation_days():
    e = P["evaluation"]
    assert not set(e["episodes_used_for_selection"]) & set(e["episodes_reserved_for_evaluation"])
    for a in LEARNED:
        assert P["learned"][a]["tried"]["episodes"] == e["episodes_used_for_selection"]


@pytest.mark.parametrize("name", RULE)
def test_rule_based_kwargs_are_arguments_of_the_strategy(name):
    r = P["rule_based"][name]
    params = inspect.signature(rx.strategy_class(r["strategy"]).__init__).parameters
    assert set(r["kwargs"]) <= set(params)
    rx.strategy_class(r["strategy"])(**r["kwargs"])          # 作れること


@pytest.mark.parametrize("algo", LEARNED)
def test_learned_entry_matches_the_recorded_models(algo):
    e = P["learned"][algo]
    assert e["strategy"] == algo and e["train"]["seeds"] == [1, 2, 3] and sorted(e["model_dirs"]) == ["1", "2", "3"]
    assert (REPO_ROOT / e["train"]["spec"]).is_file()
    for seed, d in e["model_dirs"].items():
        m = json.loads((REPO_ROOT / d / "meta.json").read_text(encoding="utf-8"))
        assert m["status"] == "ok" and m["algo"] == algo and m["seed"] == int(seed)
        assert m["step_num"] == e["train"]["step_num"] and m.get("threads", 1) == e["train"]["threads"]
        assert (m.get("train_kwargs") or {}) == e["train"]["other"]
        assert m["github_dirty"] is False and m["weights_sha1"] == e["weights_sha1"][seed]
        assert (REPO_ROOT / d / "normalize_dict.pkl").is_file()


@pytest.mark.parametrize("algo", LEARNED)
def test_weights_are_present_and_match_when_tracked(algo):
    e = P["learned"][algo]
    for seed, d in e["model_dirs"].items():
        w = REPO_ROOT / d / tm.ALGOS[algo][2]
        if not w.is_file():
            assert e["weights_in_git"] is False, f"{w} が無い（git に入っているはずの重み）"
            pytest.skip("重みが無い端末（50MB を超えるので git に入れていない）。train.spec で学習し直す")
        assert tm.weights_sha1(w) == e["weights_sha1"][seed]


@pytest.mark.parametrize("algo", LEARNED)
def test_tried_result_passed_the_preregistered_criteria(algo):
    t = P["learned"][algo]["tried"]
    assert t["passed"] is True and len(t["score_mean_by_seed"]) == 3 and min(t["score_mean_by_seed"]) > 0
    mean = sum(t["score_mean_by_seed"]) / 3
    assert (max(t["score_mean_by_seed"]) - min(t["score_mean_by_seed"])) / mean <= 0.10 + 1e-3     # 四捨五入した値からの再計算
