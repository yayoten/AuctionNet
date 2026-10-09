"""テスト共通の補助関数。pytest のフィクスチャ以外から import できるようにここに置く。"""
import gin
import numpy as np

RUN = "github.run.run_test.run_test"
CTRL = "github.simul_bidding_env.Controller.Controller.Controller"
ENV = "github.simul_bidding_env.Environment.BiddingEnv.BiddingEnv"


def gin_bindings(pv_num=500000, num_episode=2, num_tick=48, generate_log=False,
                 reserve_pv_price=0.0001, min_remaining_budget=0.1,
                 num_agent_category=8, num_category=6, pv_generator_type="neuripsPvGen") -> str:
    """config/test.gin と同じ内容を、`github.` を付けた完全修飾名で書いた gin 文字列。

    既定値は本家 config/test.gin と同じ（test_04 で一致を確認）。テストでは pv_num などを小さくして使う。
    """
    return f"""
{RUN}.generate_log = {generate_log}
{RUN}.num_episode = {num_episode}
{RUN}.num_tick = {num_tick}
{ENV}.reserve_pv_price = {reserve_pv_price}
{ENV}.min_remaining_budget = {min_remaining_budget}
{CTRL}.num_agent_category = {num_agent_category}
{CTRL}.num_category = {num_category}
{CTRL}.num_tick = {num_tick}
{CTRL}.pv_num = {pv_num}
{CTRL}.pv_generator_type = "{pv_generator_type}"
"""


def apply_gin(**kwargs) -> None:
    gin.parse_config(gin_bindings(**kwargs))


def make_history(num_ticks: int, num_pv: int, seed: int = 0, num_slots: int = 3):
    """bidding() に渡す履歴（1エージェント分）を合成する。

    形は run_test.py の `[x[i] for x in history_*]` と同じ:
      historyPValueInfo[t]       : (num_pv, 2)  [pValue, sigma]
      historyBid[t]              : (num_pv,)
      historyAuctionResult[t]    : (num_pv, 3)  [xi, slot, cost]
      historyImpressionResult[t] : (num_pv, 2)  [is_exposed, conversion]
      historyLeastWinningCost[t] : (num_pv,)
    """
    rng = np.random.default_rng(seed)
    pinfo, bids, auc, imp, lwc = [], [], [], [], []
    for _ in range(num_ticks):
        p = rng.uniform(0, 0.01, num_pv)
        sig = p * rng.uniform(0, 0.3, num_pv)
        bid = rng.uniform(0, 0.5, num_pv)
        xi = (rng.random(num_pv) < 0.3).astype(float)
        slot = np.where(xi > 0, rng.integers(1, num_slots + 1, num_pv), 0).astype(float)
        cost = np.where(xi > 0, rng.uniform(0.01, 0.3, num_pv), 0.0)
        exp = np.where(xi > 0, (rng.random(num_pv) < 0.8), 0).astype(float)
        conv = np.where(exp > 0, (rng.random(num_pv) < 0.05), 0).astype(float)
        pinfo.append(np.stack([p, sig], axis=-1))
        bids.append(bid)
        auc.append(np.stack([xi, slot, cost], axis=-1))
        imp.append(np.stack([exp, conv], axis=-1))
        lwc.append(rng.uniform(0.01, 0.3, num_pv))
    return pinfo, bids, auc, imp, lwc


def make_pvalues(num_pv: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0001, 0.01, num_pv)
    return p, p * rng.uniform(0, 0.3, num_pv)

