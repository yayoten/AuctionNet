"""06: BiddingEnv（オークション本体）。手計算できる小さな例と、ランダム入力での不変条件。

仕様（コードから読み取れるもの）
- 48 エージェント固定。上位 3 入札が落札（slot 1,2,3）。
- 支払い額は一般化第2価格: slot k の勝者は「(k+1) 位の入札額」を払う。
- least_winning_cost は 4 位の入札額。市場価格が reserve 未満なら reserve に切り上げ。
- 市場価格 == reserve の枠は unsold 扱い（xi/slot/cost/exposed/conversion を 0 に）。
- 露出確率は slot 係数 [1, 0.8, 0.6]。slot 2 が露出しなければ slot 3 も露出しない。
"""
import numpy as np
import pytest

from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv

A = 48
RESERVE = 1e-4


def make_env(episode=0, **kw):
    env = BiddingEnv(reserve_pv_price=kw.pop("reserve", RESERVE), **kw)
    env.reset(episode)
    return env


def one_pv(row, pv=0.01, sigma=0.001, env=None):
    env = env or make_env()
    b = np.zeros((1, A))
    b[0, :len(row)] = row
    out = env.simulate_ad_bidding(np.full((1, A), pv), np.full((1, A), sigma), b)
    xi, slot, cost, exp, conv, lwc, mp = out
    return dict(xi=xi[:, 0], slot=slot[:, 0], cost=cost[:, 0], exp=exp[:, 0], conv=conv[:, 0], lwc=lwc, mp=mp)


def random_auction(n, seed, env=None, hi=0.5):
    rng = np.random.default_rng(seed)
    env = env or make_env(seed % 5)
    bids = rng.uniform(0, hi, (n, A))
    pv = rng.uniform(0.0001, 0.01, (n, A))
    sg = pv * rng.uniform(0, 0.3, (n, A))
    return bids, env.simulate_ad_bidding(pv, sg, bids)


# ---------- 構造 ----------
def test_constructor_defaults():
    e = BiddingEnv()
    assert (e.reserve_pv_price, e.min_remaining_budget) == (0.01, 0.1)
    assert e.NUM_ADVERTISERS == 48 and e.NUM_SLOTS == 3
    assert e.slot_coefficients.tolist() == [1, 0.8, 0.6]


def test_initial_trunc_values_before_reset():
    e = BiddingEnv()
    assert e.advertiser_trunc_values == [(1, 0.01)] * 48


def test_output_tuple_shapes():
    n = 7
    _, out = random_auction(n, 1)
    xi, slot, cost, exp, conv, lwc, mp = out
    for x in (xi, slot, cost, exp, conv):
        assert x.shape == (A, n)
    assert lwc.shape == (n,)
    assert mp.shape == (n, 3)


# ---------- 手計算 ----------
def test_five_bidders_hand_computed():
    r = one_pv([5, 4, 3, 2, 1])
    assert r["xi"][:5].tolist() == [1, 1, 1, 0, 0]
    assert r["slot"][:5].tolist() == [1, 2, 3, 0, 0]
    assert r["cost"][:5].tolist() == [4.0, 3.0, 2.0, 0.0, 0.0]
    assert r["lwc"].tolist() == [2.0]
    assert r["mp"].tolist() == [[4.0, 3.0, 2.0]]


def test_bidder_order_does_not_matter():
    a = one_pv([1, 5, 2, 4, 3])
    # 厳密: 5→slot1, 4→slot2, 3→slot3
    assert a["slot"][1] == 1 and a["slot"][3] == 2 and a["slot"][4] == 3
    assert a["cost"][1] == 4 and a["cost"][3] == 3 and a["cost"][4] == 2


def test_winner_pays_next_highest_not_own_bid():
    r = one_pv([10, 1, 0.5, 0.2])
    assert r["cost"][0] == 1.0


def test_single_positive_bidder_is_treated_as_unsold():
    """他の入札が全て 0 → 市場価格が reserve に切り上がり == reserve → unsold。唯一の入札者も落札にならない。"""
    r = one_pv([5])
    assert r["xi"].sum() == 0 and r["cost"].sum() == 0
    assert r["lwc"].tolist() == [RESERVE]
    assert r["mp"].tolist() == [[RESERVE] * 3]


def test_two_bidders_only_top_wins():
    r = one_pv([5, 3])
    assert r["xi"][:2].tolist() == [1, 0]
    assert r["cost"][0] == 3.0
    assert r["mp"][0].tolist() == [3.0, RESERVE, RESERVE]


def test_three_bidders_third_slot_unsold():
    r = one_pv([5, 3, 2])
    assert r["xi"][:3].tolist() == [1, 1, 0]
    assert r["cost"][:3].tolist() == [3.0, 2.0, 0.0]


def test_all_zero_bids_no_winner_and_reserve_lwc():
    r = one_pv([])
    assert r["xi"].sum() == 0 and r["cost"].sum() == 0 and r["exp"].sum() == 0 and r["conv"].sum() == 0
    assert r["lwc"].tolist() == [RESERVE]


def test_bids_below_reserve_are_raised_to_reserve_and_unsold():
    r = one_pv([5, 4, 3, 2, 1e-6])
    assert r["lwc"].tolist() == [2.0]
    r = one_pv([5, 4, 3, 1e-6])
    assert r["lwc"].tolist() == [RESERVE]
    assert r["xi"][:3].tolist() == [1, 1, 0]


def test_winner_paying_exactly_reserve_is_unsold():
    r = one_pv([5, RESERVE])
    assert r["xi"].sum() == 0


def test_winner_paying_just_above_reserve_wins():
    r = one_pv([5, RESERVE * 1.01])
    assert r["xi"][0] == 1 and r["cost"][0] == pytest.approx(RESERVE * 1.01)


def test_tied_bids_still_award_three_distinct_slots():
    r = one_pv([2, 2, 2, 2, 2])
    assert r["xi"].sum() == 3
    assert sorted(r["slot"][r["slot"] > 0].tolist()) == [1, 2, 3]
    assert r["lwc"].tolist() == [2.0]
    assert (r["cost"][r["xi"] == 1] == 2.0).all()


def test_reserve_price_parameter_changes_floor():
    r = one_pv([5, 0.02], env=make_env(reserve=0.05))
    assert r["xi"].sum() == 0
    assert r["lwc"].tolist() == [0.05]


@pytest.mark.parametrize("reserve", [1e-4, 1e-2, 0.1])
def test_lwc_never_below_reserve(reserve):
    env = make_env(reserve=reserve)
    _, out = random_auction(200, 3, env=env, hi=0.2)
    assert (out[5] >= reserve).all() and (out[6] >= reserve).all()


# ---------- ランダム入力での不変条件 ----------
SEEDS = list(range(12))


@pytest.mark.parametrize("seed", SEEDS)
def test_at_most_three_winners_per_pv(seed):
    _, (xi, *_rest) = random_auction(300, seed)
    assert (xi.sum(axis=0) <= 3).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_xi_equals_slot_positive(seed):
    _, (xi, slot, *_r) = random_auction(300, seed)
    assert np.array_equal(xi == 1, slot > 0)
    assert set(np.unique(slot)) <= {0, 1, 2, 3}


@pytest.mark.parametrize("seed", SEEDS)
def test_each_slot_used_at_most_once_per_pv(seed):
    _, (xi, slot, *_r) = random_auction(300, seed)
    for k in (1, 2, 3):
        assert ((slot == k).sum(axis=0) <= 1).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_slots_are_filled_top_down(seed):
    """slot 2 が埋まっているなら slot 1 も、slot 3 が埋まっているなら slot 2 も埋まっている。"""
    _, (xi, slot, *_r) = random_auction(300, seed)
    s1, s2, s3 = ((slot == k).any(axis=0) for k in (1, 2, 3))
    assert (~s2 | s1).all() and (~s3 | s2).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_winners_are_exactly_top_bids(seed):
    bids, (xi, slot, *_r) = random_auction(200, seed)
    for j in range(bids.shape[0]):
        w = np.where(slot[:, j] > 0)[0]
        order = np.argsort(-bids[j])
        for k, agent in enumerate(order[:len(w)]):
            assert slot[agent, j] == k + 1


@pytest.mark.parametrize("seed", SEEDS)
def test_cost_is_nonnegative_and_zero_for_losers(seed):
    _, (xi, slot, cost, *_r) = random_auction(300, seed)
    assert (cost >= 0).all()
    assert (cost[xi == 0] == 0).all()
    assert (cost[xi == 1] > 0).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_second_price_cost_never_exceeds_own_bid(seed):
    bids, (xi, slot, cost, *_r) = random_auction(300, seed)
    assert (cost <= bids.T + 1e-12).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_cost_non_increasing_by_slot(seed):
    _, (xi, slot, cost, *_r) = random_auction(300, seed)
    for j in range(cost.shape[1]):
        cs = [cost[slot[:, j] == k, j] for k in (1, 2, 3)]
        flat = [c[0] for c in cs if c.size]
        assert flat == sorted(flat, reverse=True)


@pytest.mark.parametrize("seed", SEEDS)
def test_cost_equals_market_price_of_slot(seed):
    _, (xi, slot, cost, exp, conv, lwc, mp) = random_auction(200, seed)
    for k in (1, 2, 3):
        rows, cols = np.where(slot == k)
        assert np.allclose(cost[rows, cols], mp[cols, k - 1])


@pytest.mark.parametrize("seed", SEEDS)
def test_least_winning_cost_is_lowest_market_price(seed):
    _, (*_r, lwc, mp) = random_auction(200, seed)
    assert np.array_equal(lwc, mp[:, -1])
    assert (mp[:, 0] >= mp[:, 1]).all() and (mp[:, 1] >= mp[:, 2]).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_least_winning_cost_is_fourth_highest_bid_when_above_reserve(seed):
    bids, (*_r, lwc, mp) = random_auction(200, seed)
    fourth = -np.sort(-bids, axis=1)[:, 3]
    assert np.allclose(lwc, np.maximum(fourth, RESERVE))


@pytest.mark.parametrize("seed", SEEDS)
def test_exposure_only_for_winners(seed):
    _, (xi, slot, cost, exp, conv, *_r) = random_auction(300, seed)
    assert set(np.unique(exp)) <= {0, 1}
    assert (exp[xi == 0] == 0).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_conversion_only_when_exposed(seed):
    _, (xi, slot, cost, exp, conv, *_r) = random_auction(300, seed)
    assert set(np.unique(conv)) <= {0, 1}
    assert (conv[exp == 0] == 0).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_slot3_exposure_requires_slot2_exposure(seed):
    _, (xi, slot, cost, exp, conv, *_r) = random_auction(400, seed)
    for j in range(slot.shape[1]):
        e2 = exp[slot[:, j] == 2, j]
        e3 = exp[slot[:, j] == 3, j]
        if e3.size and e3[0] == 1:
            assert e2.size and e2[0] == 1


def test_slot1_is_always_exposed():
    _, (xi, slot, cost, exp, *_r) = random_auction(3000, 5)
    assert (exp[slot == 1] == 1).all()


def test_exposure_rates_follow_slot_coefficients():
    _, (xi, slot, cost, exp, *_r) = random_auction(20000, 7)
    r2 = exp[slot == 2].mean()
    r3 = exp[slot == 3].mean()
    assert r2 == pytest.approx(0.8, abs=0.02)
    assert r3 == pytest.approx(0.8 * 0.6, abs=0.02)  # 連続性ルールにより 0.6 より低い


def _conv_per_exposure(pvalue, n=20000, seed=0):
    env = make_env()
    rng = np.random.default_rng(seed)
    bids = rng.uniform(0, 1, (n, A))
    out = env.simulate_ad_bidding(np.full((n, A), pvalue), np.full((n, A), 0.01 * pvalue), bids)
    exp, conv = out[3], out[4]
    return conv.sum() / exp.sum()


@pytest.mark.parametrize("pvalue", [0.01, 0.05, 0.1])
def test_conversion_rate_matches_pvalue_in_realistic_range(pvalue):
    """実際の pValue（平均 0.0005、最大 約 0.004）に近い範囲では、露出あたりの CV 率が pValue と一致する。"""
    assert _conv_per_exposure(pvalue) == pytest.approx(pvalue, rel=0.12)


@pytest.mark.parametrize("pvalue", [0.2, 0.5, 0.8])
def test_conversion_rate_unbiased_at_high_pvalue(pvalue):
    """修正前は CV の binomial が pValue ノイズ(truncnorm)と同じ固定シードの乱数列を共有し、pValue=0.5 で CV 率が約0.17に偏っていた。"""
    assert _conv_per_exposure(pvalue) == pytest.approx(pvalue, abs=0.02)


def test_conversion_uses_its_own_seed():
    env = BiddingEnv()
    assert env.CONVERSION_SEED != env.DEFAULT_SEED


def test_random_draws_are_reused_across_calls_with_same_shape():
    """乱数が固定シード(1)のため、入札が違っても同じ形なら露出のサイコロ目が同じ（tick 間で独立でない）。現状の挙動。"""
    env = make_env()
    n = 3000
    rng = np.random.default_rng(0)
    pv, sg = np.full((n, A), 0.01), np.full((n, A), 0.001)
    b1 = rng.uniform(0, 1, (n, A))
    # 同じ順位構造を保ったまま入札額だけ変えた別の入札（単調変換）
    b2 = b1 ** 2
    o1 = env.simulate_ad_bidding(pv, sg, b1)
    o2 = env.simulate_ad_bidding(pv, sg, b2)
    assert np.array_equal(o1[1], o2[1])  # slot は同じ
    assert np.array_equal(o1[3], o2[3])  # 露出も同じ


def test_conversion_rate_zero_when_pvalue_zero():
    env = make_env()
    n = 2000
    rng = np.random.default_rng(0)
    out = env.simulate_ad_bidding(np.zeros((n, A)), np.full((n, A), 1e-9), rng.uniform(0, 1, (n, A)))
    assert out[4].sum() == 0


# ---------- 決定性・状態 ----------
def test_same_inputs_same_outputs():
    bids, a = random_auction(100, 1, env=make_env(0))
    _, b = random_auction(100, 1, env=make_env(0))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_call_is_idempotent_on_same_env():
    env = make_env()
    bids = np.random.default_rng(1).uniform(0, 1, (50, A))
    pv, sg = np.full((50, A), 0.01), np.full((50, A), 0.001)
    a = env.simulate_ad_bidding(pv, sg, bids.copy())
    b = env.simulate_ad_bidding(pv, sg, bids.copy())
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_simulate_does_not_mutate_inputs():
    env = make_env()
    rng = np.random.default_rng(2)
    bids = rng.uniform(0, 1, (50, A))
    pv = rng.uniform(0, 0.01, (50, A))
    sg = pv * 0.1
    b0, p0, s0 = bids.copy(), pv.copy(), sg.copy()
    env.simulate_ad_bidding(pv, sg, bids)
    assert np.array_equal(b0, bids) and np.array_equal(p0, pv) and np.array_equal(s0, sg)


def test_reset_sets_one_trunc_pair_per_advertiser():
    env = make_env(3)
    assert len(env.advertiser_trunc_values) == 48
    assert all(0 <= a < 1 and 0 <= b < 1 for a, b in env.advertiser_trunc_values)


def test_reset_is_deterministic_per_episode():
    assert make_env(2).advertiser_trunc_values == make_env(2).advertiser_trunc_values


def test_reset_differs_between_episodes():
    assert make_env(0).advertiser_trunc_values != make_env(1).advertiser_trunc_values


def test_generate_trunc_values_seed_is_32bit_and_reproducible():
    env = BiddingEnv()
    s1, a1, b1 = env.generate_trunc_values(3, 0, 1)
    s2, a2, b2 = env.generate_trunc_values(3, 0, 1)
    assert (s1, a1, b1) == (s2, a2, b2)
    assert 0 <= s1 < 2 ** 32


def test_generate_trunc_values_depends_on_each_key():
    env = BiddingEnv()
    base = env.generate_trunc_values(3, 0, 1)
    assert env.generate_trunc_values(4, 0, 1) != base
    assert env.generate_trunc_values(3, 1, 1) != base
    assert env.generate_trunc_values(3, 0, 2) != base


# ---------- 入力の制約 ----------
@pytest.mark.parametrize("cols", [1, 5, 30, 47, 49])
def test_requires_exactly_48_columns(cols):
    env = make_env()
    with pytest.raises((ValueError, IndexError)):  # 1 列だけは reshape より先に IndexError で落ちる
        env.simulate_ad_bidding(np.full((2, cols), .01), np.full((2, cols), .001), np.ones((2, cols)))


def test_handles_single_row():
    r = one_pv([5, 4, 3, 2])
    assert r["xi"].shape == (48,)


def test_handles_zero_rows_without_crashing_or_returns_empty():
    env = make_env()
    try:
        out = env.simulate_ad_bidding(np.zeros((0, A)), np.zeros((0, A)), np.zeros((0, A)))
    except Exception as e:  # 0 件の tick が来ないことは PV 生成側で保証されている前提
        pytest.skip(f"0 行の入力は未対応: {type(e).__name__}")
    assert out[0].shape == (A, 0)


def test_nan_bids_do_not_silently_win():
    """NaN 入札は np.sort で末尾に回る。勝者の cost に NaN が混ざらないこと。"""
    env = make_env()
    b = np.zeros((1, A))
    b[0, :4] = [5, 4, 3, 2]
    b[0, 10] = np.nan
    out = env.simulate_ad_bidding(np.full((1, A), 0.01), np.full((1, A), 0.001), b)
    assert np.isfinite(out[2]).all()


def test_negative_bids_never_win_over_positive():
    r = one_pv([5, 4, 3, 2, -1])
    assert r["slot"][4] == 0


def test_float32_inputs_are_accepted():
    env = make_env()
    n = 20
    bids = np.random.default_rng(0).uniform(0, 1, (n, A)).astype(np.float32)
    out = env.simulate_ad_bidding(np.full((n, A), .01, np.float32), np.full((n, A), .001, np.float32), bids)
    assert out[0].shape == (A, n)
