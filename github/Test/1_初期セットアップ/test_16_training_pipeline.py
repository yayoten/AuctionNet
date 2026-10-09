"""16: 学習パイプラインの通し試験。

  シミュレータ（generate_log=True）→ 生ログCSV → TrainDataGenerator → 学習スクリプト(run_*) → saved_model
  → 学習側の戦略クラスで読み込み → オフライン評価（run_evaluate）

すべて tmp ディレクトリの中で行う（学習スクリプトは cwd 相対で data/ と saved_model/ を読み書きするため、
cwd を tmp に切り替える）。本家の github/ 配下は汚さない。
学習ステップ数は、本家の既定（BC/IQL は 20000 など）では長すぎるので、テスト側で縮める。
"""
import ast
import builtins
import functools
import importlib
import logging
import pickle
import re

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import GITHUB_DIR
from helpers import apply_gin, make_history, make_pvalues

pytestmark = pytest.mark.slow

PV = 1500
NEPISODE = 2
STRAT = "github.strategy_train_env.bidding_train_env.strategy"
RUN = "github.strategy_train_env.run"


# ---------- 準備: 生ログ → 学習データ ----------
@pytest.fixture(scope="module")
def work(tmp_path_factory):
    import gin
    import github.run.run_test as rt
    from github.strategy_train_env.bidding_train_env.train_data_generator.train_data_generator import \
        TrainDataGenerator

    root = tmp_path_factory.mktemp("pipeline")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        apply_gin(pv_num=PV, num_episode=NEPISODE, num_tick=48, generate_log=True)
        try:
            rt.run_test(player_index=0)
        finally:
            gin.clear_config()
        (root / "data" / "traffic").mkdir(parents=True)
        for e in range(NEPISODE):
            (root / "data" / "log" / f"{e}.csv").rename(root / "data" / "traffic" / f"period-{e + 7}.csv")
        TrainDataGenerator(file_folder_path="./data/traffic").batch_generate_train_data()
    return root


@pytest.fixture(scope="module")
def raw(work):
    return [pd.read_csv(work / "data" / "traffic" / f"period-{e + 7}.csv") for e in range(NEPISODE)]


@pytest.fixture(scope="module")
def rl(work):
    df = pd.read_csv(work / "data/traffic/training_data_rlData_folder/training_data_all-rlData.csv")
    df["state"] = df["state"].apply(ast.literal_eval)
    df["next_state"] = df["next_state"].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else None)
    return df


# ---------- TrainDataGenerator ----------
def test_generator_creates_per_period_and_combined_files(work):
    d = work / "data/traffic/training_data_rlData_folder"
    assert (d / "period-7-rlData.csv").is_file() and (d / "period-8-rlData.csv").is_file()
    assert (d / "training_data_all-rlData.csv").is_file()


def test_generator_row_count_is_period_agent_tick(rl):
    assert len(rl) == NEPISODE * 48 * 48


def test_generator_columns(rl):
    assert list(rl.columns) == ["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget",
                                "CPAConstraint", "realAllCost", "realAllConversion", "timeStepIndex", "state",
                                "action", "reward", "reward_continuous", "done", "next_state"]


def test_state_has_16_dimensions(rl):
    assert (rl["state"].apply(len) == 16).all()


def test_first_tick_state_hand_checked(rl):
    s = rl[rl.timeStepIndex == 0]["state"]
    assert all(x[0] == 1.0 and x[1] == 1.0 for x in s)  # time_left=1, budget_left=1
    assert all(x[2:12] == (0.0,) * 10 for x in s)  # 履歴由来の特徴は 0


def test_time_left_formula(rl):
    for t in (0, 10, 47):
        assert all(x[0] == pytest.approx((48 - t) / 48) for x in rl[rl.timeStepIndex == t]["state"])


def test_budget_left_in_unit_interval_and_nonincreasing(rl):
    for _, g in rl.groupby(["deliveryPeriodIndex", "advertiserNumber"]):
        b = [s[1] for s in g.sort_values("timeStepIndex")["state"]]
        assert all(0 <= x <= 1 + 1e-9 for x in b)
        assert all(b[i + 1] <= b[i] + 1e-9 for i in range(len(b) - 1))


def test_done_flag_set_on_last_tick_for_every_trajectory(rl):
    last = rl[rl.timeStepIndex == 47]
    assert (last.done == 1).all()


def test_next_state_is_none_exactly_when_done(rl):
    assert ((rl.done == 1) == rl["next_state"].isna()).all()


def test_next_state_equals_following_tick_state(rl):
    for _, g in rl.groupby(["deliveryPeriodIndex", "advertiserNumber"]):
        g = g.sort_values("timeStepIndex")
        st, ns = g["state"].tolist(), g["next_state"].tolist()
        for i in range(len(g) - 1):
            if ns[i] is not None:
                assert ns[i] == st[i + 1]


def test_volume_features_match_raw_log(rl, raw):
    for e, df in enumerate(raw):
        vol = df[df.advertiserNumber == 0].groupby("timeStepIndex").size().to_dict()
        g = rl[(rl.deliveryPeriodIndex == e) & (rl.advertiserNumber == 0)].sort_values("timeStepIndex")
        for t, s in zip(g.timeStepIndex.astype(int), g["state"]):
            assert s[13] == vol[t]
            assert s[15] == sum(vol[k] for k in range(t))
            assert s[14] == sum(vol[k] for k in range(max(0, t - 3), t))


def test_reward_equals_conversions_of_exposed_wins_in_raw_log(rl, raw):
    for e, df in enumerate(raw):
        exposed = df[df.isExposed == 1]
        expect = exposed.groupby(["advertiserNumber", "timeStepIndex"]).conversionAction.sum()
        g = rl[rl.deliveryPeriodIndex == e].set_index(["advertiserNumber", "timeStepIndex"]).reward
        for key, val in expect.items():
            assert g.loc[key] == val


def test_real_all_conversion_is_sum_of_rewards(rl):
    for _, g in rl.groupby(["deliveryPeriodIndex", "advertiserNumber"]):
        assert g.realAllConversion.iloc[0] == g.reward.sum()


def test_action_is_total_bid_over_total_pvalue(rl, raw):
    df = raw[0]
    g = rl[(rl.deliveryPeriodIndex == 0) & (rl.advertiserNumber == 5)].set_index("timeStepIndex").action
    r = df[df.advertiserNumber == 5].groupby("timeStepIndex").agg(b=("bid", "sum"), v=("pValue", "sum"))
    for t in (0, 5, 20):
        expect = r.b[t] / r.v[t] if r.v[t] > 0 else 0
        assert g[t] == pytest.approx(expect, rel=1e-6, abs=1e-9)


def test_pid_player_alpha_at_tick0_is_15(rl):
    assert rl[(rl.advertiserNumber == 0) & (rl.timeStepIndex == 0)].action.iloc[0] == pytest.approx(15.0)


def test_budget_and_cpa_columns_match_controller(rl):
    g = rl.groupby("advertiserNumber").first()
    assert g.budget.iloc[0] == 2900 and g.CPAConstraint.iloc[0] == 100
    assert g.advertiserCategoryIndex.tolist() == [i // 8 for i in range(48)]


def test_no_nans_except_terminal_next_state(rl):
    cols = [c for c in rl.columns if c != "next_state"]
    assert not rl[cols].isna().any().any()


# ---------- 学習スクリプト ----------
def point_strategy_to(mp, module_name, root):
    """戦略クラスは「自分のファイルの 2 つ上のディレクトリ/saved_model」を読む。__file__ を tmp 配下に見せかける。"""
    mod = importlib.import_module(f"{STRAT}.{module_name}")
    mp.setattr(mod, "__file__", str(root / "x" / "y" / f"{module_name}.py"))
    return mod


def train(work, runner, *, shrink_range_in=None, patch=None):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work)
        if shrink_range_in is not None:
            mod = importlib.import_module(shrink_range_in)
            mp.setattr(mod, "range", lambda n: builtins.range(min(n, 200)), raising=False)
        if patch:
            patch(mp)
        runner()


def saved(work, d):
    return work / "saved_model" / d


def check_saved(work, d, model_file):
    p = saved(work, d)
    assert (p / model_file).is_file() and (p / "normalize_dict.pkl").is_file()
    nd = pickle.load(open(p / "normalize_dict.pkl", "rb"))
    assert set(nd) == {13, 14, 15}
    assert all(v["min"] <= v["max"] for v in nd.values())
    m = torch.jit.load(str(p / model_file))
    out = m(torch.zeros(16))
    assert out.numel() == 1 and torch.isfinite(out).all()
    return nd


@pytest.fixture(scope="module")
def trained_bc(work):
    mod = importlib.import_module(f"{RUN}.run_bc")
    train(work, mod.train_model, shrink_range_in=f"{RUN}.run_bc")
    return work


@pytest.fixture(scope="module")
def trained_iql(work):
    mod = importlib.import_module(f"{RUN}.run_iql")
    train(work, functools.partial(mod.train_iql_model, step_num=100))  # 既定は 20000。引数で縮める（test_18）
    return work


@pytest.fixture(scope="module")
def trained_cql(work):
    mod = importlib.import_module(f"{RUN}.run_cql")
    train(work, mod.train_cql_model)
    return work


@pytest.fixture(scope="module")
def trained_bcq(work):
    mod = importlib.import_module(f"{RUN}.run_bcq")
    train(work, mod.train_bcq_model)
    return work


@pytest.fixture(scope="module")
def trained_td3(work):
    mod = importlib.import_module(f"{RUN}.run_td3_bc")
    train(work, mod.train_td3_bc_model)
    return work


def test_bc_training_saves_model_and_normalize_dict(trained_bc):
    check_saved(trained_bc, "BCtest", "bc_model.pth")


def test_iql_training_saves_model_and_normalize_dict(trained_iql):
    check_saved(trained_iql, "IQLtest", "iql_model.pth")


def test_cql_training_saves_model_and_normalize_dict(trained_cql):
    check_saved(trained_cql, "CQLtest", "cql_model.pth")


def test_bcq_training_saves_model_and_normalize_dict(trained_bcq):
    check_saved(trained_bcq, "BCQtest", "bcq_model.pth")


def test_td3_bc_training_saves_model_and_normalize_dict(trained_td3):
    check_saved(trained_td3, "TD3_bctest", "td3_bc_model.pth")


def test_normalize_dict_covers_only_the_volume_features(trained_bc, rl):
    nd = pickle.load(open(saved(trained_bc, "BCtest") / "normalize_dict.pkl", "rb"))
    s13 = [s[13] for s in rl["state"]]
    assert nd[13]["min"] == min(s13) and nd[13]["max"] == max(s13)


# ---------- 学習済みモデルを戦略として読む（学習側の戦略クラス） ----------
STRATEGIES = [
    ("iql_bidding_strategy", "IqlBiddingStrategy", "trained_iql"),
    ("bc_bidding_strategy", "BcBiddingStrategy", "trained_bc"),
    ("cql_bidding_strategy", "CqlBiddingStrategy", "trained_cql"),
    ("bcq_bidding_strategy", "BcqBiddingStrategy", "trained_bcq"),
    ("td3_bc_bidding_strategy", "TD3_BCBiddingStrategy", "trained_td3"),
]


@pytest.mark.parametrize("module_name,cls,fixture", STRATEGIES)
def test_trained_model_loads_as_strategy_and_bids(module_name, cls, fixture, request):
    root = request.getfixturevalue(fixture)
    with pytest.MonkeyPatch.context() as mp:
        mod = point_strategy_to(mp, module_name, root)
        s = getattr(mod, cls)()
    s.budget, s.cpa = 3000.0, 100.0
    s.reset()
    p, sg = make_pvalues(40)
    h = make_history(5, 40)
    b0 = s.bidding(0, p, sg, [], [], [], [], [])
    b5 = s.bidding(5, p, sg, *h)
    for b in (b0, b5):
        assert b.shape == (40,) and np.isfinite(b).all()


@pytest.mark.parametrize("module_name,cls,fixture", STRATEGIES)
def test_untrained_clone_cannot_load_strategy_without_saved_model(module_name, cls, fixture):
    """クローン直後（saved_model なし）では、戦略クラスの生成はモデルファイルが無いことを示して失敗する。"""
    mod = importlib.import_module(f"{STRAT}.{module_name}")
    assert not (GITHUB_DIR / "strategy_train_env" / "saved_model").exists()
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)) as ei:
        getattr(mod, cls)()
    assert "saved_model" in str(ei.value)


# ---------- オフライン評価 ----------
@pytest.mark.parametrize("module_name,cls,fixture", STRATEGIES[:2])
def test_offline_evaluation_runs_with_trained_strategy(module_name, cls, fixture, request, caplog):
    root = request.getfixturevalue(fixture)
    run_evaluate = importlib.import_module(f"{RUN}.run_evaluate")
    with pytest.MonkeyPatch.context() as mp:
        mod = point_strategy_to(mp, module_name, root)
        mp.setattr(run_evaluate, "PlayerBiddingStrategy", getattr(mod, cls))
        mp.chdir(root)
        np.random.seed(0)
        caplog.set_level(logging.INFO)
        run_evaluate.run_test()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    score = float(re.search(r"Score: ([0-9.eE+-]+)", msgs).group(1))
    assert np.isfinite(score) and score >= 0
    assert "Total Reward" in msgs and "CPA-real" in msgs


def test_offline_evaluation_fails_on_fresh_clone_without_data():
    """run_evaluate は ./data/traffic/period-7.csv を読む（クローン直後は data/ が無い）。既定の戦略は IQL。"""
    run_evaluate = importlib.import_module(f"{RUN}.run_evaluate")
    assert run_evaluate.PlayerBiddingStrategy.__name__ == "IqlBiddingStrategy"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(GITHUB_DIR.parent)  # data/ が無い場所
        with pytest.raises(FileNotFoundError):
            run_evaluate.run_test()


def test_get_score_neurips_in_offline_eval_matches_formula():
    run_evaluate = importlib.import_module(f"{RUN}.run_evaluate")
    assert run_evaluate.getScore_neurips(100, 10, 20) == 100
    assert run_evaluate.getScore_neurips(100, 40, 20) == pytest.approx(25)


# ---------- OnlineLP / Decision Transformer ----------
def test_onlinelp_training_writes_period_csv(work):
    mod = importlib.import_module(f"{RUN}.run_onlinelp")
    train(work, mod.train_onlineLpModel)
    df = pd.read_csv(saved(work, "onlineLpTest") / "period.csv")
    assert {"realCPA", "cum_cost", "timeStepIndex", "advertiserCategoryIndex"} <= set(df.columns)
    assert len(df) > 0 and df.timeStepIndex.between(0, 47).all()


def test_onlinelp_strategy_reads_trained_period_csv(work):
    from github.simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy  # noqa: F401

    mod = importlib.import_module(f"{STRAT}.onlinelp_bidding_strategy")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod, "__file__", str(work / "x" / "y" / "onlinelp_bidding_strategy.py"))
        s = mod.OnlineLpBiddingStrategy()
    s.budget, s.cpa, s.category = 3000.0, 100.0, 0
    s.reset()
    p, sg = make_pvalues(20)
    assert s.bidding(3, p, sg, [], [], [], [], []).shape == (20,)


def test_decision_transformer_training_end_to_end(work):
    mod = importlib.import_module(f"{RUN}.run_decision_transformer")
    real_sampler = mod.WeightedRandomSampler

    def small(mp):
        mp.setattr(mod, "WeightedRandomSampler",
                   lambda w, num_samples, replacement: real_sampler(w, num_samples=64, replacement=replacement))

    train(work, mod.train_model, patch=small)
    assert (saved(work, "DTtest") / "dt.pt").is_file() and (saved(work, "DTtest") / "normalize_dict.pkl").is_file()
    nd = pickle.load(open(saved(work, "DTtest") / "normalize_dict.pkl", "rb"))
    assert nd["state_mean"].shape == (16,) and nd["state_std"].shape == (16,)


# ---------- 後始末 ----------
def test_pipeline_did_not_write_into_github_tree():
    assert not (GITHUB_DIR / "strategy_train_env" / "saved_model").exists()
    assert not (GITHUB_DIR / "strategy_train_env" / "data").exists()
    assert not (GITHUB_DIR / "data").exists()
