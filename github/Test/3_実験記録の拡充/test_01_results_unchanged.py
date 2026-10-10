"""01: 記録を足しても、結果（購入・支払い・成績）が変わらないこと。

- tick_hook を渡さない run_test（記録なし）と、記録の段 basic / standard / detail / full の実行が、同じ結果になる。
- 記録は、乱数を引かない：記録つきの実行の前後で、numpy の乱数の状態が、記録なしの実行と同じ。
- 既存の run（記録の形式の版 1）との照合は、test_05（DB に相手の run があるときだけ）。
"""
import gin
import numpy as np
import pandas as pd
import pytest

import run_experiment as rx
from conftest import GITHUB_DIR, PV, run_one

pytestmark = pytest.mark.slow
KEYS = ["reward", "allCost", "cpa", "score", "allWinPv", "budget_consumer_ratio", "second_price_ratio", "bidMean"]


def plain_run_test(strategy, player_index=3, episodes=(0, 1)):
    """run_experiment を通さず、本家の run_test を tick_hook なしで呼ぶ。結果の表と、終わったあとの numpy の乱数の次の値。"""
    from github.run.run_test import run_test
    gin.clear_config()
    gin.parse_config_files_and_bindings([str(GITHUB_DIR / "config" / "test.gin")], [f"PVNUM = {PV}"])
    np.random.seed(1)
    res = run_test(player_index=player_index, player_agent=rx.strategy_class(strategy)(), episode_ids=list(episodes))
    return pd.DataFrame(res["rawData"]).sort_values("episode").reset_index(drop=True), np.random.random()


@pytest.mark.parametrize("strategy", ["PID", "ABid", "OnlineLP", "IQL"])
def test_recorded_run_equals_run_without_hook(strategy, tmp_path, full_runs):
    rx._setup_imports()
    plain, _ = plain_run_test(strategy)
    m, d = full_runs[strategy]
    e = pd.read_parquet(d / "episodes.parquet").sort_values("episode").reset_index(drop=True)
    for a, b in [("reward", "reward"), ("allCost", "all_cost"), ("cpa", "real_cpa"), ("score", "score_component"),
                 ("allWinPv", "all_win_pv"), ("bidMean", "bid_mean"), ("second_price_ratio", "second_price_ratio")]:
        assert plain[a].tolist() == e[b].tolist(), (strategy, a)      # 丸めの差も許さない（完全一致）
    assert plain.allCost.sum() > 0 or strategy != "PID"                # PID は、縮小設定でも落札する（空の比較でないこと）


@pytest.mark.parametrize("record", ["basic", "standard", "detail"])
def test_every_record_level_gives_the_same_result(record, tmp_path, full_runs):
    m_full, d_full = full_runs["PID"]
    m, d = run_one(tmp_path, "PID", record)
    assert m["run_id"] == m_full["run_id"]                             # 記録の段は、run_id に入らない
    a, b = pd.read_parquet(d / "episodes.parquet"), pd.read_parquet(d_full / "episodes.parquet")
    for col in ["reward", "all_cost", "real_cpa", "score_component", "n_est", "n_real", "real_var"]:
        assert a[col].tolist() == b[col].tolist(), col
    ta, tb = pd.read_parquet(d / "ticks.parquet"), pd.read_parquet(d_full / "ticks.parquet")
    for col in ["cost", "reward", "alpha_eff", "n_won", "n_exposed", "est_sum_exposed", "real_sum_exposed"]:
        assert ta[col].tolist() == tb[col].tolist(), col


def test_recording_draws_no_random_numbers(tmp_path):
    """記録つきの run_test のあとの numpy の乱数の状態が、記録なしのときと同じ（記録が、全体の乱数を 1 つも引いていない）。"""
    rx._setup_imports()
    from github.run.run_test import run_test
    _, after_plain = plain_run_test("PID")
    seen = []
    gin.clear_config()
    gin.parse_config_files_and_bindings([str(GITHUB_DIR / "config" / "test.gin")], [f"PVNUM = {PV}"])
    np.random.seed(1)
    run_test(player_index=3, player_agent=rx.strategy_class("PID")(), episode_ids=[0, 1], tick_hook=seen.append)
    assert np.random.random() == after_plain
    assert len(seen) == 96 and {"values_real", "conversion_draw", "exposure_draw", "sim_iters", "internals"} <= set(seen[0])
