"""19: BCQ の学習の入口（run_bcq.train_bcq_model）に出した `max_action` の検証。

- 既定値は、従来の直書きの値（BCQ クラスの既定の 100）。渡さないときと、100 を渡したときで、重みが一致する。
- 渡すと、保存した重みの生成モデルと方策の上限が、その値になる（出力がその範囲に収まる）。
"""
import inspect
import random

import numpy as np
import torch

from test_18_model_dir_and_train_args import rl_csv  # noqa: F401  （合成の学習データ）
from github.strategy_train_env.bidding_train_env.baseline.bcq.bcq import BCQ
from github.strategy_train_env.run import run_bcq


def _train(rl_csv, out, **kw):
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    run_bcq.train_bcq_model(train_data_path=str(rl_csv), save_path=str(out), step_num=5, **kw)
    return torch.jit.load(str(out / "bcq_model.pth"))


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_default_is_the_old_hardcoded_value():
    assert inspect.signature(run_bcq.train_bcq_model).parameters["max_action"].default == 100
    assert inspect.signature(BCQ.__init__).parameters["max_action"].default == 100


def test_omitting_max_action_gives_the_same_weights_as_100(rl_csv, tmp_path):
    assert _same(_train(rl_csv, tmp_path / "a"), _train(rl_csv, tmp_path / "b", max_action=100))


def test_given_max_action_is_used_by_generator_and_actor(rl_csv, tmp_path):
    m100, m300 = _train(rl_csv, tmp_path / "a"), _train(rl_csv, tmp_path / "b", max_action=300)
    assert (m100.vae.max_action, m100.actor.max_action) == (100, 100)
    assert (m300.vae.max_action, m300.actor.max_action) == (300, 300)
    assert not _same(m100, m300)
    # 生成モデルの最後の層に大きな値を入れると、候補は上限に張り付く。方策の摂動は上限の 5%（phi）までなので、
    # 出力は「上限の 95%〜上限」に入る（上限が効いていることの確認）
    for m, cap in ((m100, 100.0), (m300, 300.0)):
        with torch.no_grad():
            m.vae.d3.bias.fill_(50.0)
            torch.manual_seed(0)
            assert cap * 0.95 - 1e-3 <= float(m(torch.zeros(16))) <= cap + 1e-3
