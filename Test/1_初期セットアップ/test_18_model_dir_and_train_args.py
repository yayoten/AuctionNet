"""18: 学習ベース手法を「自前で学習して、シミュレータで評価する」ために引数へ出した値の検証。

- シミュレータ側の学習ベース戦略（BC/IQL/CQL/BCQ/TD3_BC）の `model_dir`：既定（None）は同梱の重みで従来どおり。
  指定すると、そのフォルダの重みを読む。
- 学習スクリプト（run_*.py）の `train_data_path` / `save_path` / `step_num`：既定値は従来の直書きの値。
  指定すると、その場所から読み、その場所へ書き、その回数だけ学習する。
"""
import importlib
import inspect
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import SIM_DIR
from helpers import make_history, make_pvalues
from github.simul_bidding_env.strategy.bc_bidding_strategy import BcBiddingStrategy
from github.simul_bidding_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
from github.simul_bidding_env.strategy.cql_bidding_strategy import CqlBiddingStrategy
from github.simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from github.simul_bidding_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy

OFFICIAL = SIM_DIR / "strategy" / "official_agent"
RUN = "github.strategy_train_env.run"
# key: (戦略クラス, 同梱フォルダ, 重みのファイル名, 学習モジュール, 学習関数, 既定のステップ数)
ALGOS = {
    "bc": (BcBiddingStrategy, "BCtest", "bc_model.pth", "run_bc", "train_model", 20000),
    "iql": (IqlBiddingStrategy, "IQLtest", "iql_model.pth", "run_iql", "train_iql_model", 20000),
    "cql": (CqlBiddingStrategy, "CQLtest", "cql_model.pth", "run_cql", "train_cql_model", 100),
    "bcq": (BcqBiddingStrategy, "BCQtest", "bcq_model.pth", "run_bcq", "train_bcq_model", 100),
    "td3_bc": (TD3_BCBiddingStrategy, "TD3_bctest", "td3_bc_model.pth", "run_td3_bc", "train_td3_bc_model", 100),
}
OLD_DATA_PATH = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"


@pytest.fixture(params=list(ALGOS))
def key(request):
    return request.param


def bids_of(strategy, tick=5):
    strategy.budget, strategy.cpa = 3000.0, 100.0
    strategy.reset()
    p, sg = make_pvalues(50)
    torch.manual_seed(1)  # BCQ は推論で乱数を引く
    return np.asarray(strategy.bidding(tick, p, sg, *make_history(tick, 50)))


# ---------- シミュレータ側：model_dir ----------
def test_model_dir_default_is_none_and_points_to_bundled_weights(key):
    cls, d = ALGOS[key][0], ALGOS[key][1]
    assert inspect.signature(cls.__init__).parameters["model_dir"].default is None
    assert cls().model_dir == str(OFFICIAL / d)


def test_model_dir_copy_of_bundled_gives_identical_bids(key, tmp_path):
    cls, d = ALGOS[key][0], ALGOS[key][1]
    shutil.copytree(OFFICIAL / d, tmp_path / "m")
    np.testing.assert_array_equal(bids_of(cls(model_dir=str(tmp_path / "m"))), bids_of(cls()))


def test_model_dir_is_actually_read(tmp_path):
    """BC の戦略に IQL の重みを（BC のファイル名で）渡すと、同梱の IQL と同じ入札になり、同梱の BC とは違う。"""
    m = tmp_path / "m"
    m.mkdir()
    shutil.copy(OFFICIAL / "IQLtest" / "iql_model.pth", m / "bc_model.pth")
    shutil.copy(OFFICIAL / "IQLtest" / "normalize_dict.pkl", m / "normalize_dict.pkl")
    got = bids_of(BcBiddingStrategy(model_dir=str(m)))
    np.testing.assert_allclose(got, bids_of(IqlBiddingStrategy()), rtol=1e-6)
    assert not np.allclose(got, bids_of(BcBiddingStrategy()))


def test_model_dir_missing_raises(key, tmp_path):
    with pytest.raises(Exception):
        ALGOS[key][0](model_dir=str(tmp_path / "nothing"))


# ---------- 学習スクリプト：train_data_path / save_path / step_num ----------
def test_train_function_defaults_are_the_old_hardcoded_values(key):
    _, d, _, mod, fn, steps = ALGOS[key]
    sig = inspect.signature(getattr(importlib.import_module(f"{RUN}.{mod}"), fn)).parameters
    assert sig["train_data_path"].default == OLD_DATA_PATH
    assert sig["save_path"].default == f"saved_model/{d}"
    assert sig["step_num"].default == steps


@pytest.fixture(scope="module")
def rl_csv(tmp_path_factory):
    """学習データ（rlData）と同じ列・同じ書式の、小さな合成データ。4 本の軌跡 × 48 ティック。"""
    rng = np.random.default_rng(0)
    rows = []
    for traj in range(4):
        states = [tuple(float(x) for x in np.r_[(48 - t) / 48, 1 - t / 60, rng.uniform(0, 1, 11), rng.integers(100, 5000, 3)])
                  for t in range(48)]
        for t in range(48):
            rows.append(dict(deliveryPeriodIndex=7, advertiserNumber=traj, advertiserCategoryIndex=0, budget=3000.0,
                             CPAConstraint=100.0, realAllCost=2000.0, realAllConversion=20.0, timeStepIndex=t,
                             state=states[t], action=float(rng.uniform(20, 150)), reward=float(rng.integers(0, 3)),
                             reward_continuous=float(rng.uniform(0, 2)), done=int(t == 47),
                             next_state=states[t + 1] if t < 47 else None))
    path = tmp_path_factory.mktemp("rl") / "any_name.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _count_steps(mp, module):
    """学習の 1 ステップ（model.step）が何回呼ばれたかを数える。"""
    calls = []
    for name in ("BC", "IQL", "CQL", "BCQ", "TD3_BC"):
        cls = getattr(module, name, None)
        if cls is not None:
            orig = cls.step
            mp.setattr(cls, "step", lambda self, *a, _o=orig, **k: (calls.append(1), _o(self, *a, **k))[1])
    return calls


def test_train_reads_given_path_writes_given_dir_and_runs_given_steps(key, rl_csv, tmp_path):
    cls, _, model_file, mod, fn, _ = ALGOS[key]
    module = importlib.import_module(f"{RUN}.{mod}")
    out = tmp_path / "out" / "model"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)  # 既定の ./data も ./saved_model も無い場所
        calls = _count_steps(mp, module)
        getattr(module, fn)(train_data_path=str(rl_csv), save_path=str(out), step_num=7)
    assert len(calls) == 7
    assert (out / model_file).is_file() and (out / "normalize_dict.pkl").is_file()
    assert not (tmp_path / "saved_model").exists()
    # 保存した重みを、シミュレータ側の戦略が model_dir で読めて、有限の入札を返す
    bids = bids_of(cls(model_dir=str(out)))
    assert bids.shape == (50,) and np.isfinite(bids).all()


def test_bcq_strategy_calls_both_saved_forms(rl_csv, tmp_path):
    """同梱の BCQ は forward(states, eval_flag)、学習コードが保存する BCQ は forward(states)。戦略はどちらも呼べる。"""
    module = importlib.import_module(f"{RUN}.run_bcq")
    module.train_bcq_model(train_data_path=str(rl_csv), save_path=str(tmp_path / "m"), step_num=3)
    assert BcqBiddingStrategy()._takes_eval_flag is True
    assert BcqBiddingStrategy(model_dir=str(tmp_path / "m"))._takes_eval_flag is False
