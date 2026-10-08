"""04: 評価の実行スクリプト（research/src/run_experiment.py）が、学習ベースの戦略を扱えること。

- 同梱の重み（model_dir なし）と、自前で学習した重み（model_dir あり）の両方を、縮小設定（PV 数 3000）で通す。
- ルールベース（PID）が、これまでどおり動く（実行スクリプトを書き換えたことの回帰）。
出力先は tmp に差し替える（DB/runs には書かない）。
"""
import json

import pandas as pd
import pytest

import run_experiment as rx
import train_model as tm

pytestmark = pytest.mark.slow
LEARNED = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "DB_RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


@pytest.fixture(scope="module")
def trained(tmp_path_factory, rl_csv):
    """5 手法を、合成データで数ステップだけ学習した重み。{手法: フォルダ}"""
    root = tmp_path_factory.mktemp("models")
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tm, "MODELS", root)
        for algo in LEARNED:
            p = root / f"{algo}.json"
            p.write_text(json.dumps(dict(name="t", algo=algo, step_num=8, train_data=str(rl_csv))), encoding="utf-8")
            (task,), _ = tm.build_tasks(str(p), False)
            m = tm.execute(task)
            assert m["status"] == "ok", m.get("traceback")
            out[algo] = root / m["model_id"]
    return out


def run(tmp_path, strategy, kwargs=None, episodes=(0,)):
    spec = dict(name="t", player=dict(strategy=strategy, kwargs=kwargs or {}), player_indices=[3], episodes=list(episodes),
                gin={"PVNUM": 3000})
    p = tmp_path / f"spec_{strategy}_{len(list(tmp_path.glob('spec_*.json')))}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    tasks, _ = rx.build_tasks(str(p), False)
    assert len(tasks) == 1
    return rx.execute_run(tasks[0])


def test_all_eight_strategies_are_registered():
    assert set(rx.STRATEGIES) == {"PID", "ABid", "OnlineLP", *LEARNED}
    for name in rx.STRATEGIES:
        assert rx.strategy_class(name).__name__ == rx.STRATEGIES[name][1]


def test_rule_based_still_runs(tmp_path, runs):
    m = run(tmp_path, "PID")
    assert m["status"] == "ok", m.get("traceback")
    assert m["model_id"] is None and m["model_dir"] is None


@pytest.mark.parametrize("algo", LEARNED)
def test_bundled_weights_run(algo, tmp_path, runs):
    m = run(tmp_path, algo)
    assert m["status"] == "ok", m.get("traceback")
    assert m["strategy"] == algo and m["model_id"] is None
    ticks = pd.read_parquet(runs / m["run_id"] / "ticks.parquet")
    assert len(ticks) == 48 and (ticks.alpha_eff >= 0).all() and ticks.alpha_eff.notna().all()


@pytest.mark.parametrize("algo", LEARNED)
def test_self_trained_weights_run_and_are_recorded(algo, tmp_path, runs, trained):
    m = run(tmp_path, algo, {"model_dir": str(trained[algo])})
    assert m["status"] == "ok", m.get("traceback")
    assert m["model_id"] == trained[algo].name
    assert m["model_sha1"] == json.loads((trained[algo] / "meta.json").read_text(encoding="utf-8"))["model_sha1"]
    assert m["model_step_num"] == 8
    params = json.loads((runs / m["run_id"] / "params.json").read_text(encoding="utf-8"))
    assert params["effective"]["player"]["kwargs"]["model_dir"] == str(trained[algo])
    ep = pd.read_parquet(runs / m["run_id"] / "episodes.parquet")
    assert len(ep) == 1 and ep.score_component.notna().all()


def test_run_id_separates_bundled_and_self_trained(tmp_path, runs, trained):
    a, b = run(tmp_path, "IQL"), run(tmp_path, "IQL", {"model_dir": str(trained["IQL"])})
    assert a["run_id"] != b["run_id"]


def test_model_dir_is_resolved_from_repo_root():
    """spec にはリポジトリ直下からの相対パスで書く。絶対パスもそのまま通る。"""
    assert rx.make_strategy({"strategy": "IQL", "kwargs": {}}).model_dir.endswith("official_agent/IQLtest")
    rel = "github/simul_bidding_env/strategy/official_agent/IQLtest"
    assert rx.make_strategy({"strategy": "IQL", "kwargs": {"model_dir": rel}}).model_dir == str(rx.REPO / rel)


def test_checkpoint_dir_can_be_evaluated_and_keeps_its_own_id(tmp_path, runs, rl_csv, monkeypatch):
    monkeypatch.setattr(tm, "MODELS", tmp_path / "models")
    p = tmp_path / "t.json"
    p.write_text(json.dumps(dict(name="t", algo="IQL", step_num=8, checkpoints=[4], train_data=str(rl_csv))), encoding="utf-8")
    (task,), _ = tm.build_tasks(str(p), False)
    m = tm.execute(task)
    ck = tmp_path / "models" / m["model_id"] / "ckpt" / f"{m['model_id']}_s000004"
    r = run(tmp_path, "IQL", {"model_dir": str(ck)})
    assert r["status"] == "ok", r.get("traceback")
    assert r["model_id"] == f"{m['model_id']}_s000004" and r["model_step_num"] == 4
