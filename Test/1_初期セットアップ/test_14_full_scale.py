"""14: README の手順そのまま（`cd github && python main_test.py`）で、本家設定の評価が最後まで動くこと。

- 設定は config/test.gin のまま: PVNUM=500000・48 エージェント・2 エピソード・プレイヤー 0 と 1。
- PYTHONPATH は渡さない（main_test.py が自分でリポジトリ直下を sys.path に入れる）。
- クローン直後は学習済みモデルが無いので、プレイヤーは PID にフォールバックする。
- 約 100 秒かかる。`pytest -m "not full_scale"` で除外できる。
- 基準値は golden_full_scale.json。取得した OS/CPU（"platform"）と違う端末では、乱数・浮動小数の実装差で
  ずれうるので、基準値との比較だけスキップする（動作そのものの検証は全部走る）。
- その端末の基準値を作る／更新するとき:  UPDATE_GOLDEN=1 python -m pytest test_14_full_scale.py
"""
import ast
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pytest

import conftest
from conftest import GITHUB_DIR, HERE

pytestmark = [pytest.mark.slow, pytest.mark.full_scale]

import platform

GOLDEN_PATH = HERE / "golden_full_scale.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
THIS_PLATFORM = f"{platform.system()}-{platform.machine()}"
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN") == "1"


@pytest.fixture
def golden(full):
    """基準値。UPDATE_GOLDEN=1 なら今回の結果で書き換える。取得元と違うプラットフォームでは比較をスキップ。"""
    global GOLDEN
    if UPDATE_GOLDEN and GOLDEN.get("_updated_in_this_run") is None:
        assert full.returncode == 0, full.stderr[-800:]
        GOLDEN = {
            "_provenance": f"UPDATE_GOLDEN=1 で {THIS_PLATFORM} / Python {sys.version.split()[0]} 上で取得。",
            "platform": THIS_PLATFORM, "python": "%d.%d" % sys.version_info[:2],
            "rank_score": full.rank_score, "analysis_info": full.info,
        }
        GOLDEN_PATH.write_text(json.dumps(GOLDEN, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        GOLDEN["_updated_in_this_run"] = True
    if GOLDEN.get("platform") != THIS_PLATFORM:
        pytest.skip(f"基準値は {GOLDEN.get('platform')} で取得。この端末は {THIS_PLATFORM}。"
                    "UPDATE_GOLDEN=1 で、この端末の基準値を作れる")
    return GOLDEN


class Full:
    pass


@pytest.fixture(scope="module")
def full():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    t = time.time()
    r = subprocess.run([sys.executable, "-W", "ignore", "main_test.py"], cwd=GITHUB_DIR, env=env,
                       capture_output=True, text=True, timeout=850)
    F = Full()
    F.elapsed = time.time() - t
    F.returncode, F.stdout, F.stderr = r.returncode, r.stdout, r.stderr
    m = re.search(r"rank_score:\s*([0-9.eE+-]+)", r.stdout)
    F.rank_score = float(m.group(1)) if m else None
    m = re.search(r"analysis_info\s*(\{.*\})", r.stdout)
    F.info = ast.literal_eval(m.group(1)) if m else None
    return F


def test_main_test_py_exits_zero_as_documented_in_readme(full):
    assert full.returncode == 0, full.stderr[-1500:]


def test_prints_rank_score_and_analysis_info(full):
    assert full.rank_score is not None and full.info is not None


def test_no_gin_ambiguity_error(full):
    assert "Ambiguous selector" not in full.stderr
    assert "Traceback" not in full.stderr


def test_two_players_were_evaluated(full):
    assert "PlayerIndex:0" in full.stderr and "PlayerIndex:1" in full.stderr
    assert all(len(v) == 2 for v in full.info.values())


def test_two_episodes_per_player(full):
    assert len(re.findall(r"PlayerIndex:0 episode:\d evaluate", full.stderr)) == 2
    assert len(re.findall(r"PlayerIndex:1 episode:\d evaluate", full.stderr)) == 2


def test_player_falls_back_to_pid_on_fresh_clone(full):
    assert "Failed to load PlayerAgent" in full.stderr
    assert "playerAgentName:PidBiddingStrategy0" in full.stderr


def test_player_budget_and_cpa_logged_match_controller_tables(full):
    assert "playerAgentCpa:100 playerAgentBudget:2900 playerAgentCategory:0 PlayerIndex:0" in full.stderr
    assert "playerAgentCpa:70 playerAgentBudget:4350 playerAgentCategory:0 PlayerIndex:1" in full.stderr


def test_runtime_is_reasonable(full):
    assert full.elapsed < 800


def test_analysis_info_keys(full):
    assert set(full.info) == {"category", "reward", "win_pv_ratio", "budget_consumer_ratio", "second_price_ratio",
                              "last_compete_tick_index", "cpa_exceedance_Rate"}


def test_rank_score_matches_golden(full, golden):
    assert full.rank_score == pytest.approx(golden["rank_score"], rel=1e-6)


@pytest.mark.parametrize("key", ["category", "reward", "last_compete_tick_index"])
def test_integer_metrics_match_golden_exactly(full, golden, key):
    assert full.info[key] == golden["analysis_info"][key]


@pytest.mark.parametrize("key", ["win_pv_ratio", "budget_consumer_ratio", "second_price_ratio", "cpa_exceedance_Rate"])
def test_ratio_metrics_match_golden(full, golden, key):
    assert full.info[key] == pytest.approx(golden["analysis_info"][key], abs=2e-3)


def test_budget_is_never_exceeded(full):
    assert all(0 < r <= 1 for r in full.info["budget_consumer_ratio"])


def test_scores_are_nonnegative_and_finite(full):
    assert full.rank_score >= 0 and np.isfinite(full.rank_score)


def test_second_price_ratio_is_below_one(full):
    """第 2 価格オークションなので、支払い/落札額（second_price_ratio）は 1 以下。"""
    assert all(0 < r <= 1 for r in full.info["second_price_ratio"])


def test_win_ratio_is_a_small_positive_fraction(full):
    """48 体で 3 枠を争う。1 体の勝率は 0 より大きく、3/48 の数倍以内。"""
    assert all(0 < r < 0.3 for r in full.info["win_pv_ratio"])


def test_github_tree_was_not_modified_by_the_run(full):
    assert conftest._github_status() == conftest.GITHUB_STATUS_AT_START


@pytest.mark.parametrize("rel", ["data", "saved_model", "strategy_train_env/saved_model", "strategy_train_env/data"])
def test_run_leaves_no_output_dirs(full, rel):
    assert not (GITHUB_DIR / rel).exists()


def test_run_is_reproducible_within_this_machine(full):
    """基準値の有無に関係なく、同じ端末で 2 回回して同じ結果になること（プラットフォームに依存しない検証）。"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-W", "ignore", "main_test.py"], cwd=GITHUB_DIR, env=env,
                       capture_output=True, text=True, timeout=850)
    assert r.returncode == 0, r.stderr[-800:]
    again = float(re.search(r"rank_score:\s*([0-9.eE+-]+)", r.stdout).group(1))
    assert again == full.rank_score
