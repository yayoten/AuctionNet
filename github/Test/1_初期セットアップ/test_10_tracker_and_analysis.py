"""10: BiddingTracker（ログ→学習用CSV）と PlayerAnalysis（スコア計算）。"""
import numpy as np
import pandas as pd
import pytest

from github.simul_bidding_env.Tracker.BiddingTracker import BiddingTracker
from github.simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis

COLUMNS = ["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint",
           "timeStepIndex", "remainingBudget", "pvIndex", "pValue", "pValueSigma", "bid", "xi", "adSlot", "cost",
           "isExposed", "conversionAction", "leastWinningCost", "isEnd"]


# ---------- BiddingTracker ----------
def log_tick(tr, episode, tick, n_pv=3, n_agent=4, total=0, seed=0):
    rng = np.random.default_rng(seed + tick)
    pv = rng.uniform(0, 1, (n_pv, n_agent))
    sg = pv * 0.1
    bids = rng.uniform(0, 1, (n_pv, n_agent))
    xi = (rng.random((n_agent, n_pv)) < .5).astype(int)
    slot = xi * rng.integers(1, 4, (n_agent, n_pv))
    cost = xi * rng.uniform(0, 1, (n_agent, n_pv))
    exp = xi
    conv = (rng.random((n_agent, n_pv)) < .3).astype(int) * exp
    lwc = rng.uniform(0, 1, n_pv)
    done = np.zeros(n_agent, dtype=int)
    budgets = np.array([100., 200., 300., 400.])[:n_agent]
    cpa = np.array([10., 20., 30., 40.])[:n_agent]
    cat = np.array([0, 0, 1, 1])[:n_agent]
    rem = budgets - tick
    tr.train_logging(episode, tick, pv, budgets, cpa, cat, rem, total, sg, bids, xi, slot, cost, exp, conv, lwc, done)
    return dict(pv=pv, sg=sg, bids=bids, xi=xi, slot=slot, cost=cost, exp=exp, conv=conv, lwc=lwc, budgets=budgets,
                cpa=cpa, cat=cat, rem=rem)


def test_tracker_starts_empty():
    tr = BiddingTracker()
    assert tr.pValues == [] and tr.bids == [] and tr.isEnds == []


def test_tracker_reset_clears_everything():
    tr = BiddingTracker()
    log_tick(tr, 0, 0)
    tr.reset()
    assert all(len(getattr(tr, a)) == 0 for a in vars(tr) if isinstance(getattr(tr, a), list))


def test_tracker_default_name():
    assert BiddingTracker().name == "BiddingTracker"
    assert BiddingTracker("x").name == "x"


def test_logging_appends_one_block_per_call():
    tr = BiddingTracker()
    for t in range(3):
        log_tick(tr, 0, t)
    assert len(tr.pValues) == 3 and all(len(a) == 12 for a in tr.pValues)


def test_generate_csv_has_expected_columns_and_row_count(tmp_path):
    tr = BiddingTracker()
    for t in range(2):
        log_tick(tr, 0, t)
    path = tmp_path / "sub" / "log.csv"
    tr.generate_train_data(str(path))
    df = pd.read_csv(path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2 * 3 * 4


def test_generate_csv_creates_parent_directories(tmp_path):
    tr = BiddingTracker()
    log_tick(tr, 0, 0)
    p = tmp_path / "a" / "b" / "c.csv"
    tr.generate_train_data(str(p))
    assert p.is_file()


def test_generate_csv_values_align_with_inputs(tmp_path):
    tr = BiddingTracker()
    d = log_tick(tr, 1, 5, n_pv=3, n_agent=4, total=100)
    p = tmp_path / "l.csv"
    tr.generate_train_data(str(p))
    df = pd.read_csv(p)
    # 行順: pv-major, agent-minor
    for pv_i in range(3):
        for a in range(4):
            row = df.iloc[pv_i * 4 + a]
            assert row.deliveryPeriodIndex == 1 and row.timeStepIndex == 5
            assert row.advertiserNumber == a
            assert row.advertiserCategoryIndex == d["cat"][a]
            assert row.budget == d["budgets"][a] and row.CPAConstraint == d["cpa"][a]
            assert row.remainingBudget == pytest.approx(d["rem"][a])
            assert row.pvIndex == 100 + pv_i
            assert row.pValue == pytest.approx(d["pv"][pv_i, a])
            assert row.pValueSigma == pytest.approx(d["sg"][pv_i, a])
            assert row.bid == pytest.approx(d["bids"][pv_i, a])
            assert row.xi == d["xi"][a, pv_i] and row.adSlot == d["slot"][a, pv_i]
            assert row.cost == pytest.approx(d["cost"][a, pv_i])
            assert row.isExposed == d["exp"][a, pv_i] and row.conversionAction == d["conv"][a, pv_i]
            assert row.leastWinningCost == pytest.approx(d["lwc"][pv_i])


def test_generate_csv_marks_done_flag(tmp_path):
    tr = BiddingTracker()
    rng = np.random.default_rng(0)
    n_pv, n_a = 2, 3
    tr.train_logging(0, 47, rng.random((n_pv, n_a)), np.ones(n_a), np.ones(n_a), np.zeros(n_a), np.ones(n_a), 0,
                     rng.random((n_pv, n_a)), rng.random((n_pv, n_a)), np.zeros((n_a, n_pv)), np.zeros((n_a, n_pv)),
                     np.zeros((n_a, n_pv)), np.zeros((n_a, n_pv)), np.zeros((n_a, n_pv)), np.zeros(n_pv),
                     np.array([0, 1, 1]))
    p = tmp_path / "d.csv"
    tr.generate_train_data(str(p))
    df = pd.read_csv(p)
    assert df.groupby("advertiserNumber").isEnd.first().tolist() == [0, 1, 1]


def test_generate_without_logging_raises(tmp_path):
    with pytest.raises(ValueError):
        BiddingTracker().generate_train_data(str(tmp_path / "x.csv"))


def test_tracker_csv_is_readable_by_training_data_generator(tmp_path):
    """列名が TrainDataGenerator の要求（groupby 等）を満たす。"""
    tr = BiddingTracker()
    for t in range(3):
        log_tick(tr, 0, t)
    p = tmp_path / "x.csv"
    tr.generate_train_data(str(p))
    df = pd.read_csv(p)
    for c in ["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint",
              "timeStepIndex", "bid", "leastWinningCost", "conversionAction", "xi", "pValue", "isExposed", "cost",
              "isEnd", "remainingBudget"]:
        assert c in df.columns


# ---------- PlayerAnalysis ----------
def feed(pa, episode=0, player=0, cpa=20.0, budget=100.0, values=(1, 2), costs=(10, 20), compete=(100, 100),
         win=(10, 20), allwin=(5.0, 6.0), bid_mean=(0.1, 0.2)):
    for t, (v, c, cp, w, aw, bm) in enumerate(zip(values, costs, compete, win, allwin, bid_mean)):
        pa.logging_player_tick(episode, t, player, cpa, budget, v, c, cp, w, aw, bm)


def analyze(pa, name="p", idx=0, cat=0):
    pa.player_multi_episode(name)
    return pa.get_return_res(name, idx, cat)


def test_analysis_default_state_empty():
    pa = PlayerAnalysis()
    assert pa.multi_episode_data == [] and pa.analysis_res == []


def test_analysis_basic_metrics_hand_computed():
    pa = PlayerAnalysis()
    feed(pa)  # reward=3, cost=30 → real CPA=10 ≤ 20 → penalty なし
    r = analyze(pa)
    assert r["reward"] == 3
    assert r["allCost"] == 30
    assert r["cpa"] == pytest.approx(10.0)
    assert r["targetValue"] == pytest.approx(60.0)
    assert r["budget_consumer_ratio"] == pytest.approx(0.3)
    assert r["win_pv_ratio"] == pytest.approx(30 / 200)
    assert r["second_price_ratio"] == pytest.approx(30 / 11.0)
    assert r["cpa_exceedance_Rate"] == pytest.approx((10 - 20) / 20)
    assert r["last_compete_tick_index"] == 1
    assert r["bidMean"] == pytest.approx(0.15)
    assert r["score"] == pytest.approx(3 / 20000)


def test_analysis_penalty_applies_when_cpa_exceeds_constraint():
    pa = PlayerAnalysis()
    feed(pa, cpa=5.0)  # real CPA = 10 > 5 → (5/10)^2 = 0.25
    r = analyze(pa)
    assert r["score"] == pytest.approx(3 * 0.25 / 20000)


def test_analysis_no_penalty_at_exactly_the_constraint():
    pa = PlayerAnalysis()
    feed(pa, cpa=10.0)
    assert analyze(pa)["score"] == pytest.approx(3 / 20000)


def test_get_score_neurips_formula():
    pa = PlayerAnalysis()
    assert pa._get_score_neurips(100, 10, 20) == 100
    assert pa._get_score_neurips(100, 40, 20) == pytest.approx(25.0)
    assert pa._get_score_neurips(0, 40, 20) == 0
    assert pa._get_score_neurips(100, 20, 20) == 100


def test_analysis_zero_reward_does_not_divide_by_zero():
    pa = PlayerAnalysis()
    feed(pa, values=(0, 0), costs=(0, 0), win=(0, 0))
    r = analyze(pa)
    assert r["reward"] == 0 and r["score"] == 0
    assert np.isfinite(r["cpa"]) and r["last_compete_tick_index"] == -1


def test_analysis_cost_without_reward_gets_huge_cpa_and_zero_score():
    pa = PlayerAnalysis()
    feed(pa, values=(0, 0), costs=(10, 10), win=(1, 1))
    r = analyze(pa)
    assert r["cpa"] > 1e9 and r["score"] == pytest.approx(0.0, abs=1e-6)


def test_multi_episode_reward_is_summed_and_other_metrics_averaged():
    pa = PlayerAnalysis()
    feed(pa, episode=0, values=(1, 2), costs=(10, 20))
    feed(pa, episode=1, values=(3, 4), costs=(5, 5))
    r = analyze(pa)
    assert r["reward"] == 10
    assert r["allCost"] == pytest.approx((30 + 10) / 2)
    assert len(r["rawData"]) == 2


def test_multi_episode_score_sums_over_episodes():
    pa = PlayerAnalysis()
    feed(pa, episode=0, values=(1, 2), costs=(10, 20), cpa=20)  # score 3
    feed(pa, episode=1, values=(3, 4), costs=(5, 5), cpa=20)  # score 7
    assert analyze(pa)["score"] == pytest.approx(10 / 20000)


def test_ticks_are_sorted_before_analysis():
    pa = PlayerAnalysis()
    pa.logging_player_tick(0, 1, 0, 20, 100, 2, 20, 100, 20, 6, .2)
    pa.logging_player_tick(0, 0, 0, 20, 100, 1, 10, 100, 10, 5, .1)
    assert analyze(pa)["reward"] == 3


def test_find_last_non_zero_index():
    f = PlayerAnalysis._find_last_non_zero_index
    assert f([0, 0, 0]) == -1 and f([1, 0, 0]) == 0 and f([0, 3, 0, 2, 0]) == 3 and f([]) == -1


def test_reset_clears_analysis():
    pa = PlayerAnalysis()
    feed(pa)
    analyze(pa)
    pa.reset()
    assert pa.multi_episode_data == [] and pa.analysis_res == []


def test_return_res_metadata_fields():
    pa = PlayerAnalysis()
    feed(pa)
    r = analyze(pa, name="myagent", idx=7, cat=3)
    assert (r["policyName"], r["playerindex"], r["category"]) == ("myagent", 7, 3)
    for k in ["cpaConstraint", "budget", "reward", "allCost", "cpa", "targetValue", "allCompetePv", "allWinPv",
              "win_pv_ratio", "budget_consumer_ratio", "second_price_ratio", "cpa_exceedance_Rate",
              "last_compete_tick_index", "bidMean", "score", "rawData"]:
        assert k in r


def test_analysis_with_empty_data_returns_nan_silently():
    """データが空でも例外にならず、reward=0 / score=0 / その他の平均は NaN になる（エラーに気づきにくい挙動）。"""
    pa = PlayerAnalysis()
    pa.player_multi_episode("x")
    r = pa.get_return_res("x", 0, 0)
    assert r["reward"] == 0 and r["score"] == 0
    assert np.isnan(r["cpa"]) and np.isnan(r["allCost"])


@pytest.mark.parametrize("reward,cpa_real,cpa_c", [(10, 5, 10), (10, 10, 10), (10, 20, 10), (10, 100, 10), (1e4, 12, 10)])
def test_score_is_never_above_reward_and_nonnegative(reward, cpa_real, cpa_c):
    s = PlayerAnalysis()._get_score_neurips(reward, cpa_real, cpa_c)
    assert 0 <= s <= reward


def test_score_is_monotone_decreasing_in_cpa_above_constraint():
    pa = PlayerAnalysis()
    vals = [pa._get_score_neurips(100, c, 10) for c in (10, 11, 15, 20, 50)]
    assert vals == sorted(vals, reverse=True)
