"""13: 縮小設定（pv_num=2000）で run_test を最後まで回す統合テスト。

本家の設定どおり（PVNUM=500000）で回すテストは test_14_full_scale.py。
ここでは速く回せる設定で、結果の整合性・保存ログ・予算制約・再現性を、多面的に確かめる。
"""
import numpy as np
import pandas as pd
import pytest

import github.run.run_test as rt
from github.simul_bidding_env.Controller.Controller import Controller

PV = 2000
RESULT_KEYS = ["policyName", "playerindex", "category", "rawData", "cpaConstraint", "budget", "reward", "allCost",
               "cpa", "targetValue", "allCompetePv", "allWinPv", "win_pv_ratio", "budget_consumer_ratio",
               "second_price_ratio", "cpa_exceedance_Rate", "last_compete_tick_index", "bidMean", "score"]


def run(bind, player_index=0, num_episode=1, **kw):
    bind(pv_num=PV, num_episode=num_episode, **kw)
    return rt.run_test(player_index=player_index)


@pytest.fixture
def tiny_budget(monkeypatch):
    """全エージェントの予算を 8 に絞る。支出が予算に届き、過払い調整ループが実際に働く。"""
    monkeypatch.setattr(Controller, "calculate_budget", lambda self: [8.0] * 48)


# ---------- 実行できること・結果の形 ----------
def test_run_test_completes_and_returns_expected_keys(bind_small_gin):
    r = run(bind_small_gin)
    assert set(RESULT_KEYS) <= set(r)


def test_result_types_and_ranges(bind_small_gin):
    r = run(bind_small_gin)
    assert r["policyName"] == "PidBiddingStrategy0"
    assert r["playerindex"] == 0 and r["category"] == 0
    assert r["reward"] >= 0 and r["score"] >= 0
    assert 0 <= r["win_pv_ratio"] <= 1
    assert 0 <= r["budget_consumer_ratio"] <= 1 + 1e-9
    assert -1 <= r["last_compete_tick_index"] <= 47
    assert np.isfinite([r["cpa"], r["allCost"], r["bidMean"]]).all()


def test_rawdata_has_one_entry_per_episode(bind_small_gin):
    r = run(bind_small_gin, num_episode=2)
    assert len(r["rawData"]) == 2
    assert [x["episode"] for x in r["rawData"]] == [0, 1]


def test_player_metadata_matches_controller_tables(bind_small_gin):
    r = run(bind_small_gin, player_index=1)
    assert r["cpaConstraint"] == 70 and r["budget"] == 4350 and r["category"] == 0


@pytest.mark.parametrize("pidx", [0, 1, 8, 47])
def test_player_index_selects_budget_cpa_category(bind_small_gin, pidx):
    r = run(bind_small_gin, player_index=pidx)
    c = Controller.__new__(Controller)
    assert r["budget"] == c.calculate_budget()[pidx]
    assert r["cpaConstraint"] == c.get_cpa_constraints()[pidx]
    assert r["category"] == pidx // 8


def test_all_cost_never_exceeds_player_budget(bind_small_gin):
    r = run(bind_small_gin)
    assert r["allCost"] <= r["budget"] + 1e-6


def test_score_matches_formula(bind_small_gin):
    r = run(bind_small_gin, num_episode=2)
    expect = sum(min(1, (e["cpaConstraint"] / (e["cpa"] + 1e-10)) ** 2) * e["reward"] if e["cpa"] > e["cpaConstraint"]
                 else e["reward"] for e in r["rawData"]) / 20000
    assert r["score"] == pytest.approx(expect)


def test_reward_is_sum_over_episodes(bind_small_gin):
    r = run(bind_small_gin, num_episode=2)
    assert r["reward"] == sum(e["reward"] for e in r["rawData"])


def test_allCompetePv_matches_generated_traffic(bind_small_gin):
    r = run(bind_small_gin)
    assert PV - 48 <= r["allCompetePv"] <= PV


# ---------- 再現性 ----------
def test_two_runs_give_identical_results(bind_small_gin):
    a = run(bind_small_gin)
    b = run(bind_small_gin)
    for k in RESULT_KEYS:
        if k != "rawData":
            assert a[k] == b[k], k


def test_reproducible_after_reseeding_global_rngs(bind_small_gin):
    import torch

    torch.manual_seed(1); np.random.seed(1)
    a = run(bind_small_gin)
    torch.manual_seed(999); np.random.seed(999)
    b = run(bind_small_gin)
    assert a["reward"] == b["reward"] and a["allCost"] == b["allCost"]


def test_different_players_give_different_outcomes(bind_small_gin):
    a = run(bind_small_gin, player_index=0)
    b = run(bind_small_gin, player_index=2)
    assert (a["budget"], a["cpaConstraint"]) != (b["budget"], b["cpaConstraint"])


# ---------- 予算制約（過払い調整ループ） ----------
def test_tiny_budget_is_respected(bind_small_gin, tiny_budget):
    r = run(bind_small_gin, num_episode=2)
    assert r["budget"] == 8.0
    for e in r["rawData"]:
        assert e["allCost"] <= 8.0 + 1e-6


def test_tiny_budget_is_actually_binding(bind_small_gin, tiny_budget):
    """予算 8 では使い切る（調整ループの経路を通っている証拠）。"""
    r = run(bind_small_gin)
    assert r["budget_consumer_ratio"] > 0.9


def test_tiny_budget_agent_stops_bidding_early(bind_small_gin, tiny_budget):
    r = run(bind_small_gin)
    assert r["last_compete_tick_index"] < 47


# ---------- generate_log=True のログ ----------
class Logged:
    """generate_log=True・予算 8・2 エピソードで 1 回だけ回した結果（このモジュールの全ログ検証で共有）。"""


@pytest.fixture(scope="module")
def logged_run(tmp_path_factory):
    import gin
    from helpers import apply_gin

    out = tmp_path_factory.mktemp("logged_run")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Controller, "calculate_budget", lambda self: [8.0] * 48)
        mp.chdir(out)
        apply_gin(pv_num=PV, num_episode=2, num_tick=48, generate_log=True)
        try:
            result = rt.run_test(player_index=0)
        finally:
            gin.clear_config()
    L = Logged()
    L.dir = out
    L.result = result
    L.dfs = [pd.read_csv(out / "data" / "log" / f"{e}.csv") for e in range(2)]
    return L


@pytest.fixture
def logged(logged_run):
    return logged_run.result, logged_run.dfs


def test_log_files_are_created_in_cwd(logged_run):
    assert (logged_run.dir / "data/log/0.csv").is_file() and (logged_run.dir / "data/log/1.csv").is_file()


def test_log_columns(logged):
    from test_10_tracker_and_analysis import COLUMNS

    assert list(logged[1][0].columns) == COLUMNS


def test_log_row_count_is_pv_times_48_agents(logged):
    r, dfs = logged
    for df in dfs:
        n_pv = df.groupby("timeStepIndex").pvIndex.nunique().sum()
        assert len(df) == n_pv * 48


def test_log_total_pv_close_to_pv_num(logged):
    for df in logged[1]:
        total_pv = len(df) // 48  # 1 PV = 48 行（エージェントごと）
        assert PV - 48 <= total_pv <= PV


def test_log_each_pv_has_one_row_per_agent(logged):
    for df in logged[1]:
        assert (df.groupby(["timeStepIndex", "pvIndex"]).size() == 48).all()


def test_log_episode_and_tick_ranges(logged):
    for e, df in enumerate(logged[1]):
        assert (df.deliveryPeriodIndex == e).all()
        assert df.timeStepIndex.min() == 0 and df.timeStepIndex.max() == 47


def test_log_advertiser_attributes_match_controller(logged):
    df = logged[1][0]
    c = Controller.__new__(Controller)
    budget, cpa = c.calculate_budget(), c.get_cpa_constraints()
    g = df.groupby("advertiserNumber").first()
    assert g.index.tolist() == list(range(48))
    assert np.array_equal(g.CPAConstraint.values, cpa)
    assert np.allclose(g.budget.values, 8.0)  # tiny_budget
    assert g.advertiserCategoryIndex.tolist() == [i // 8 for i in range(48)]


def test_log_at_most_three_winners_per_pv(logged):
    for df in logged[1]:
        assert (df.groupby(["timeStepIndex", "pvIndex"]).xi.sum() <= 3).all()


def test_log_slots_are_distinct_within_pv(logged):
    for df in logged[1]:
        w = df[df.adSlot > 0]
        assert not w.duplicated(["timeStepIndex", "pvIndex", "adSlot"]).any()


def test_log_xi_matches_adslot(logged):
    for df in logged[1]:
        assert ((df.xi == 1) == (df.adSlot > 0)).all()


def test_log_conversion_only_when_exposed_and_exposed_only_when_won(logged):
    for df in logged[1]:
        assert (df[df.isExposed == 0].conversionAction == 0).all()
        assert (df[df.xi == 0].isExposed == 0).all()
        assert (df[df.xi == 0].cost == 0).all()


def test_log_total_spend_never_exceeds_budget_for_any_agent(logged):
    """48 エージェントすべてについて、露出された落札の支払い合計が予算以内（過払い調整が全員に効いている）。"""
    for df in logged[1]:
        spend = (df.cost * df.isExposed).groupby(df.advertiserNumber).sum()
        assert (spend <= 8.0 + 1e-6).all(), spend.max()


def test_log_remaining_budget_equals_budget_minus_cumulative_spend(logged):
    for df in logged[1]:
        tick_spend = (df.cost * df.isExposed).groupby([df.advertiserNumber, df.timeStepIndex]).sum().unstack()
        cum_before = tick_spend.cumsum(axis=1).shift(1, axis=1).fillna(0)
        rem = df.groupby(["advertiserNumber", "timeStepIndex"]).remainingBudget.first().unstack()
        assert np.allclose(rem.values, 8.0 - cum_before.values, atol=1e-6)


def test_log_remaining_budget_is_non_increasing(logged):
    for df in logged[1]:
        rem = df.groupby(["advertiserNumber", "timeStepIndex"]).remainingBudget.first().unstack()
        assert (np.diff(rem.values, axis=1) <= 1e-9).all()


def test_log_isend_set_on_last_tick_for_everyone(logged):
    for df in logged[1]:
        assert (df[df.timeStepIndex == 47].isEnd == 1).all()


def test_log_isend_matches_min_remaining_budget_rule(logged):
    for df in logged[1]:
        t = df[df.timeStepIndex < 47]
        assert ((t.remainingBudget < 0.1) == (t.isEnd == 1)).all()


def test_log_player_rows_reproduce_result_reward_and_cost(logged):
    r, dfs = logged
    p = [df[df.advertiserNumber == 0] for df in dfs]
    reward = sum(x.conversionAction.sum() for x in p)
    assert reward == r["reward"]
    for x, e in zip(p, r["rawData"]):
        assert (x.cost * x.isExposed).sum() == pytest.approx(e["allCost"])


def test_log_exhausted_agents_stop_bidding(logged):
    for df in logged[1]:
        t = df[(df.timeStepIndex < 47) & (df.remainingBudget < 0.1)]
        assert (t.bid == 0).all()


def test_log_least_winning_cost_is_per_pv_constant(logged):
    for df in logged[1]:
        assert (df.groupby(["timeStepIndex", "pvIndex"]).leastWinningCost.nunique() == 1).all()


def test_log_pvalues_are_probabilities(logged):
    for df in logged[1]:
        assert df.pValue.between(0, 1).all() and (df.pValueSigma >= 0).all()


def test_log_pvindex_is_unique_per_episode(logged):
    """修正前は total_pv_num が 0 のまま更新されず、pvIndex が tick ごとに 0 から振り直されていた。"""
    for df in logged[1]:
        per_pv = df.drop_duplicates(["timeStepIndex", "pvIndex"])
        assert per_pv.pvIndex.is_unique


def test_log_pvindex_is_contiguous_from_zero_and_ordered_by_tick(logged):
    for df in logged[1]:
        idx = np.sort(df.pvIndex.unique())
        assert np.array_equal(idx, np.arange(len(idx)))
        first = df.groupby("timeStepIndex").pvIndex.min()
        last = df.groupby("timeStepIndex").pvIndex.max()
        assert first.iloc[0] == 0
        assert (first.values[1:] == last.values[:-1] + 1).all()


def test_log_pvindex_restarts_each_episode(logged):
    assert all(df.pvIndex.min() == 0 for df in logged[1])


def test_second_episode_keeps_the_small_pv_num(logged):
    """reset() が pv_num を保つので、2 エピソード目も縮小設定のまま。"""
    for df in logged[1]:
        assert PV - 48 <= df.pvIndex.nunique() <= PV


def test_no_log_files_when_generate_log_false(bind_small_gin, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(bind_small_gin)
    assert not (tmp_path / "data").exists()


def test_github_directory_is_not_written_by_runs(bind_small_gin, tmp_path, monkeypatch):
    """generate_log=True の出力先は cwd 相対。cwd を tmp にすれば本家ディレクトリは汚れない。"""
    from conftest import GITHUB_DIR

    before = sorted(p.name for p in GITHUB_DIR.iterdir())
    monkeypatch.chdir(tmp_path)
    run(bind_small_gin, generate_log=True)
    assert sorted(p.name for p in GITHUB_DIR.iterdir()) == before
