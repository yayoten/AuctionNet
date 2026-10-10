"""02: 環境の乱数と、予算超過の取り消しについて、これまで「コードを読んだだけ」だったこと（REP006.md の C3・C5・C6）を、実行して確かめる。

確かめた結果は README.md の「分かったこと」に書いた。ここは、その根拠になるテストである。
"""
import numpy as np
import pytest
from scipy.stats import truncnorm

from github.run.run_test import adjust_over_cost, get_winner
from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv


def make(n, seed):
    rng = np.random.default_rng(seed)
    pv = rng.uniform(0.0001, 0.002, (n, 48))
    return pv, pv * rng.uniform(0.01, 0.3, (n, 48))


@pytest.fixture
def env():
    e = BiddingEnv()
    e.reset(episode=0)
    return e


# ---------- C3：雑音の打ち切り ----------
def test_c3_noise_is_truncated_asymmetrically_per_advertiser(env):
    pv, sg = make(4000, 0)
    z = (env._generate_values_matrix(pv, sg) - pv) / sg            # 標準化した雑音
    u = np.array(env.advertiser_trunc_values)                       # (48, 2) = (u1, u2)
    assert ((0 <= u) & (u < 1)).all()
    assert (z >= -2 * u[:, 0] - 1e-9).all() and (z <= 2 * u[:, 1] + 1e-9).all()     # 下側 −2×u1、上側 +2×u2
    assert np.abs(u[:, 0] - u[:, 1]).mean() > 0.1                                    # 上下で対称でない
    # 雑音の平均は、打ち切り正規分布の平均に一致する（広告主ごとに、正にも負にも偏る）
    want = truncnorm.mean(-2 * u[:, 0], 2 * u[:, 1])
    np.testing.assert_allclose(z.mean(axis=0), want, atol=0.03)
    assert (want > 0.05).any() and (want < -0.05).any()


def test_c3_trunc_values_depend_on_advertiser_and_episode_only(env):
    a = list(env.advertiser_trunc_values)
    env.reset(episode=0)
    assert env.advertiser_trunc_values == a                        # 同じエピソードなら同じ
    env.reset(episode=1)
    assert env.advertiser_trunc_values != a                        # エピソードで変わる
    assert len(set(a)) == 48                                       # 広告主で違う
    assert len(env.advertiser_trunc_seeds) == 48 and all(isinstance(s, int) for s in env.advertiser_trunc_seeds)


# ---------- C6：乱数が、呼ぶたびに同じ種で作り直される ----------
def test_c6_same_noise_for_the_same_opportunity_index_in_every_tick(env):
    """標準化した雑音は、(機会の番号, 広告主) だけで決まり、ティック（中身）にも、そのティックの機会の数にも依らない。"""
    (pv1, sg1), (pv2, sg2), (pv3, sg3) = make(500, 1), make(500, 2), make(300, 3)
    z1 = (env._generate_values_matrix(pv1, sg1) - pv1) / sg1
    z2 = (env._generate_values_matrix(pv2, sg2) - pv2) / sg2
    z3 = (env._generate_values_matrix(pv3, sg3) - pv3) / sg3
    np.testing.assert_allclose(z1, z2, atol=1e-9)
    np.testing.assert_allclose(z1[:300], z3, atol=1e-9)


def test_c6_the_underlying_uniform_is_shared_across_episodes(env):
    """エピソードが変わると打ち切りの幅（u1・u2）は変わるが、もとの一様乱数は同じ。"""
    pv, sg = make(500, 1)
    u0 = np.array(env.advertiser_trunc_values)
    q0 = truncnorm.cdf((env._generate_values_matrix(pv, sg) - pv) / sg, -2 * u0[:, 0], 2 * u0[:, 1])
    env.reset(episode=3)
    u3 = np.array(env.advertiser_trunc_values)
    q3 = truncnorm.cdf((env._generate_values_matrix(pv, sg) - pv) / sg, -2 * u3[:, 0], 2 * u3[:, 1])
    np.testing.assert_allclose(q0, q3, atol=1e-6)


def test_c6_conversion_draw_is_shared_across_ticks_until_a_zero_probability_shifts_the_stream(env):
    """購入の抽選も、同じ種で作り直される。確率が 0 の機会は乱数を使わないので、そこから先は並びがずれる。"""
    p = np.full((2000, 48), 0.30)
    a = env._calculate_conversion_action(p, 1)
    assert np.array_equal(a, env._calculate_conversion_action(p, 1))                 # 同じ入力なら同じ結果
    b = env._calculate_conversion_action(np.full((2000, 48), 0.31), 1)
    assert (a == b).mean() > 0.98                                                     # 独立なら約 0.58。ほぼ同じ乱数が当たっている
    shifted = p.copy()
    shifted[0, 0] = 0.0
    c = env._calculate_conversion_action(shifted, 1)
    assert (a == c).mean() < 0.65                                                     # 先頭に確率 0 が 1 つ入ると、独立と同じ程度まで崩れる


def test_c6_exposure_rates_and_slot3_rule(env):
    rng = np.random.default_rng(0)
    slot = np.zeros((6000, 48), dtype=int)
    for i in range(len(slot)):
        slot[i, rng.choice(48, 3, replace=False)] = [1, 2, 3]
    e = env._calculate_exposure(slot.copy())
    draw = env.last_record["exposure_draw"]
    assert e[slot == 1].mean() == 1.0 and abs(e[slot == 2].mean() - 0.8) < 0.03
    assert abs(draw[slot == 3].mean() - 0.6) < 0.03                 # 抽選そのものは 0.6
    assert abs(e[slot == 3].mean() - 0.48) < 0.03                   # 枠 2 が露出しないと取り消されるので、0.8 × 0.6
    assert np.array_equal(e, env._calculate_exposure(slot.copy()))  # 同じ枠の並びなら、同じ結果


# ---------- C5：予算超過の取り消し ----------
def test_c5_adjust_over_cost_drops_a_fixed_share_of_each_slots_wins_with_seed_1(env):
    pv, sg = make(3000, 5)
    bids = pv * 5000.0
    _, slot, cost_pit, exposed, *_ = env.simulate_ad_bidding(pv, sg, bids)
    winner = get_winner(slot)
    ratio = np.zeros(48)
    ratio[7] = 0.25
    before = bids.copy()
    adjust_over_cost(bids, ratio, env.slot_coefficients, winner)
    changed = np.where((before != bids).any(axis=0))[0].tolist()
    assert changed == [7]                                                            # 超過した広告主の入札だけが変わる
    for k in range(3):
        mine = np.where(winner[:, k] == 7)[0]
        dropped = mine[bids[mine, 7] == 0]
        assert len(dropped) == int(np.ceil(len(mine) * 0.25))                        # 枠ごとに、落札の ceil(数 × 超過率) を 0 にする
        want = np.random.default_rng(seed=1).choice(mine, len(dropped), replace=False)
        assert sorted(dropped) == sorted(want)                                       # 種 1 の乱数で選ぶ（毎回、同じ選び方）
    assert (bids[(slot.T == 0)[:, 7], 7] == before[(slot.T == 0)[:, 7], 7]).all()    # 落札していない機会の入札は、変わらない


# ---------- 記録用の量（last_record）が、返り値と整合する ----------
def test_last_record_is_consistent_with_the_returned_arrays(env):
    pv, sg = make(3000, 7)
    bids = pv * 5000.0
    xi, slot, cost, exposed, conv, lwc, mp = env.simulate_ad_bidding(pv, sg, bids)
    r = env.last_record
    assert np.array_equal(r["values"], env._generate_values_matrix(pv, sg))
    assert np.array_equal(conv.T, r["conversion_draw"] * exposed.T)                   # 購入 = 抽選 × 露出
    assert (exposed.T <= r["exposure_draw"]).all()                                    # 最終の露出は、抽選のあとで減るだけ
    assert set(r["seconds"]) == {"sort", "slot", "cost", "exposure", "values", "conversion", "unsold"}
    assert all(v >= 0 for v in r["seconds"].values())
    assert np.array_equal(np.sort(r["sorted_bid_indices"], axis=1), np.sort(get_winner(slot), axis=1))
