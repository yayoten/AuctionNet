"""05: 本番の学習で出来た重み（DB/models/）が、記録どおりで、シミュレータの戦略として使えること。

DB/models/ が空の端末（まだ学習していない）では、理由を表示してスキップする。
"""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import run_experiment as rx
import train_model as tm
from conftest import HERE, MODELS_DIR, REPO_ROOT

sys.path.insert(0, str(HERE.parent / "1_初期セットアップ"))
from helpers import make_history, make_pvalues  # noqa: E402

METAS = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(MODELS_DIR.glob("M*/meta.json"))] if MODELS_DIR.is_dir() else []
ALGOS = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]


@pytest.fixture(params=METAS, ids=[f"{m['algo']}-{m['model_id']}" for m in METAS])
def meta(request):
    return request.param


def test_models_exist():
    if not METAS:
        pytest.skip("DB/models が空。research/src/train_model.py で学習する")
    assert all(m["status"] == "ok" for m in METAS), [m["model_id"] for m in METAS if m["status"] != "ok"]


def test_every_algo_has_a_default_model():
    """本家の既定設定（既定のステップ数、seed=1）の重みが、5 手法ぶんある。"""
    if not METAS:
        pytest.skip("DB/models が空")
    have = {m["algo"] for m in METAS if m["step_num_is_default"] and m["seed"] == 1 and m["status"] == "ok"}
    assert have == set(ALGOS)


def test_files_and_recorded_sha1(meta):
    d = MODELS_DIR / meta["model_id"]
    assert (d / "normalize_dict.pkl").is_file() and (d / "loss.csv").is_file()
    assert tm.sha1_of(d / meta["model_file"]) == meta["model_sha1"]


def test_trained_from_committed_code_on_cpu(meta):
    assert meta["github_dirty"] is False and meta["device"] == "cpu"


def test_loss_is_finite_and_covers_every_step(meta):
    loss = pd.read_csv(MODELS_DIR / meta["model_id"] / "loss.csv")
    assert meta["loss_all_finite"] is True
    assert list(loss.step) == list(range(meta["step_num"]))


def test_strategy_loads_and_bids_are_finite(meta):
    s = rx.make_strategy({"strategy": meta["algo"], "kwargs": {"model_dir": f"DB/models/{meta['model_id']}"}})
    s.budget, s.cpa = 3000.0, 100.0
    s.reset()
    p, sg = make_pvalues(200)
    for tick in (0, 1, 10, 47):
        h = make_history(tick, 200, seed=tick) if tick else ([], [], [], [], [])
        bids = np.asarray(s.bidding(tick, p, sg, *h))
        assert bids.shape == (200,) and np.isfinite(bids).all()


@pytest.mark.needs_data
@pytest.mark.slow
@pytest.mark.parametrize("algo", ALGOS)
def test_default_model_is_reproduced_by_retraining(algo, tmp_path, monkeypatch):
    """既定設定の重みを、同じ学習データ・同じ乱数シードで学習し直すと、同じ重みになる（重みのテンソルの SHA-1 が一致）。
    保存ファイルの SHA-1（model_sha1）では比べない。torch.jit のファイルには、同じプロセスでそれまでに保存したモデルの数で
    変わる連番が入り、重みが同じでもファイルが一致しないことがある（コマンドから学習し直せば、ファイルの SHA-1 も一致する）。"""
    ms = [m for m in METAS if m["algo"] == algo and m["step_num_is_default"] and m["seed"] == 1]
    if not ms:
        pytest.skip("既定設定の重みが無い")
    m = ms[0]
    spec = tmp_path / "s.json"
    # 学習データは今の置き場所（既定の DB/dataset/…）から読む。meta の train_data は学習した時点のパスで、置き場所を移す前の
    # 記録（dataset/…）も残っている。同じデータであることは、下の SHA-1 の一致で確かめる。
    spec.write_text(json.dumps(dict(name="retrain", algo=algo)), encoding="utf-8")
    monkeypatch.setattr(tm, "MODELS", tmp_path / "models")
    (task,), _ = tm.build_tasks(str(spec), False)
    assert task["resolved"]["train_data_sha1"] == m["train_data_sha1"]
    again = tm.execute(task)
    assert again["status"] == "ok", again.get("traceback")
    new = tmp_path / "models" / again["model_id"]
    # model_id には github/ の版が入る。学習した時点から github/ を変えていれば（例：BCQ の max_action を引数に出した）、
    # ID は変わる。そのときも、重みと損失が一致すること（変更が既定の学習を変えていないこと）は、下で確かめる。
    if again["github_version"] == m["github_version"]:
        assert again["model_id"] == m["model_id"]
    assert tm.weights_sha1(new / m["model_file"]) == tm.weights_sha1(MODELS_DIR / m["model_id"] / m["model_file"])
    pd.testing.assert_frame_equal(pd.read_csv(new / "loss.csv"), pd.read_csv(MODELS_DIR / m["model_id"] / "loss.csv"))
