"""03: 学習の実行スクリプト（research/src/train_model.py）。spec.json → 学習 → DB/models/<model_id>/ への記録。

合成の小さな学習データを使い、出力先は tmp に差し替える（DB/ には書かない）。
"""
import json
import pickle

import pandas as pd
import pytest
import torch

import train_model as tm

ALGOS = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]
OFFICIAL_DEFAULT_STEPS = {"BC": 20000, "IQL": 20000, "CQL": 100, "BCQ": 100, "TD3_BC": 100}


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "MODELS", tmp_path / "models")
    return tmp_path / "models"


def write_spec(tmp_path, rl_csv, **kw):
    spec = dict(name="t", work="W001", rep="REPTEST", train_data=str(rl_csv), step_num=12, **kw)
    p = tmp_path / f"spec_{len(list(tmp_path.glob('spec_*.json')))}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def run(spec_path, force=False):
    tasks, skipped = tm.build_tasks(str(spec_path), force)
    return [tm.execute(t) for t in tasks], skipped


def test_default_step_num_is_the_official_default():
    assert {a: tm.default_step_num(a) for a in ALGOS} == OFFICIAL_DEFAULT_STEPS


def test_null_step_num_resolves_to_default_and_is_flagged(tmp_path, rl_csv, models):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(dict(name="t", algo="CQL", train_data=str(rl_csv))), encoding="utf-8")
    (task,), _ = tm.build_tasks(str(p), False)
    assert task["resolved"]["step_num"] == 100 and task["resolved"]["step_num_is_default"] is True


@pytest.mark.parametrize("algo", ALGOS)
def test_trains_and_records(algo, tmp_path, rl_csv, models):
    (m,), _ = run(write_spec(tmp_path, rl_csv, algo=algo))
    assert m["status"] == "ok", m.get("traceback")
    d = models / m["model_id"]
    assert sorted(p.name for p in d.iterdir()) == sorted(["loss.csv", "meta.json", "normalize_dict.pkl", tm.ALGOS[algo][2]])
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["algo"] == algo and meta["step_num"] == 12 and meta["step_num_is_default"] is False and meta["seed"] == 1
    assert meta["model_sha1"] == tm.sha1_of(d / tm.ALGOS[algo][2]) and meta["train_data_sha1"] == tm.sha1_of(rl_csv)
    assert meta["device"] == "cpu" and meta["loss_all_finite"] is True
    assert len(pd.read_csv(d / "loss.csv")) == 12                      # 損失は 1 ステップに 1 行
    assert set(pickle.load(open(d / "normalize_dict.pkl", "rb"))) == {13, 14, 15}
    out = torch.jit.load(str(d / tm.ALGOS[algo][2]))(torch.zeros(16))
    assert torch.isfinite(out).all()


def test_sweep_expands_to_one_model_per_point(tmp_path, rl_csv, models):
    p = write_spec(tmp_path, rl_csv, algo="BC", sweep={"algo": ["BC", "CQL"], "step_num": [5, 9]})
    tasks, _ = tm.build_tasks(str(p), False)
    assert sorted((t["resolved"]["algo"], t["resolved"]["step_num"]) for t in tasks) == [("BC", 5), ("BC", 9), ("CQL", 5), ("CQL", 9)]
    assert len({t["model_id"] for t in tasks}) == 4


@pytest.mark.parametrize("algo", ALGOS)
def test_same_spec_gives_same_weights_and_is_skipped_next_time(algo, tmp_path, rl_csv, models):
    p = write_spec(tmp_path, rl_csv, algo=algo)
    (a,), _ = run(p)
    weights_before = tm.weights_sha1(models / a["model_id"] / tm.ALGOS[algo][2])
    again, skipped = run(p)
    assert again == [] and skipped == [a["model_id"]]
    (b,), _ = run(p, force=True)
    assert b["model_id"] == a["model_id"]
    w = tm.weights_sha1(models / a["model_id"] / tm.ALGOS[algo][2])
    assert w == weights_before      # 乱数を固定しているので、学習し直しても重みが一致する


def test_seed_changes_model_id_and_weights(tmp_path, rl_csv, models):
    (a,), _ = run(write_spec(tmp_path, rl_csv, algo="IQL"))
    (b,), _ = run(write_spec(tmp_path, rl_csv, algo="IQL", seed=2))
    assert a["model_id"] != b["model_id"]
    assert tm.weights_sha1(models / a["model_id"] / "iql_model.pth") != tm.weights_sha1(models / b["model_id"] / "iql_model.pth")


def test_failure_is_recorded_not_raised(tmp_path, rl_csv, models, monkeypatch):
    p = write_spec(tmp_path, rl_csv, algo="BC")
    (task,), _ = tm.build_tasks(str(p), False)
    task["resolved"]["train_data"] = str(tmp_path / "missing.csv")
    m = tm.execute(task)
    assert m["status"] == "error" and "missing.csv" in m["error"]
    again, skipped = run(p)                                        # 失敗した学習は、次の実行で流し直される
    assert len(again) == 1 and skipped == []


# ---------- チェックポイント（学習の途中の重み）とスレッド数 ----------
@pytest.mark.parametrize("algo", ALGOS)
def test_checkpoints_are_saved_and_do_not_change_the_final_weights(algo, tmp_path, rl_csv, models):
    (plain,), _ = run(write_spec(tmp_path, rl_csv, algo=algo))
    w_plain = tm.weights_sha1(models / plain["model_id"] / tm.ALGOS[algo][2])
    (ck,), _ = run(write_spec(tmp_path, rl_csv, algo=algo, checkpoints=[4, 8, 12, 99]), force=True)
    assert ck["model_id"] == plain["model_id"]                  # checkpoints は model_id に入らない（最後の重みが同じなので）
    assert ck["checkpoints"] == [4, 8]                          # step_num（12）以上は無視する
    d = models / ck["model_id"]
    assert tm.weights_sha1(d / tm.ALGOS[algo][2]) == w_plain    # 途中で保存しても、最後の重みは変わらない
    for step in (4, 8):
        c = d / "ckpt" / f"{ck['model_id']}_s{step:06d}"
        meta = json.loads((c / "meta.json").read_text(encoding="utf-8"))
        assert meta["step_num"] == step and meta["parent_model_id"] == ck["model_id"] and meta["is_checkpoint"] is True
        assert meta["weights_sha1"] == tm.weights_sha1(c / tm.ALGOS[algo][2]) != w_plain
        assert (c / "normalize_dict.pkl").is_file()


@pytest.mark.parametrize("algo", ALGOS)
def test_checkpoint_equals_a_shorter_training(algo, tmp_path, rl_csv, models):
    """8 ステップ目のチェックポイントは、8 ステップだけ学習した重みと同じ（乱数の列が同じなので）。"""
    (ck,), _ = run(write_spec(tmp_path, rl_csv, algo=algo, checkpoints=[8]))
    spec = dict(name="t", algo=algo, step_num=8, train_data=str(rl_csv))
    p = tmp_path / "short.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    (short,), _ = run(p)
    c = models / ck["model_id"] / "ckpt" / f"{ck['model_id']}_s000008" / tm.ALGOS[algo][2]
    assert tm.weights_sha1(c) == tm.weights_sha1(models / short["model_id"] / tm.ALGOS[algo][2])


def test_threads_enter_model_id_only_when_not_one(tmp_path, rl_csv, models):
    a, _ = tm.build_tasks(str(write_spec(tmp_path, rl_csv, algo="BC")), False)
    b, _ = tm.build_tasks(str(write_spec(tmp_path, rl_csv, algo="BC", threads=1)), False)
    c, _ = tm.build_tasks(str(write_spec(tmp_path, rl_csv, algo="BC", threads=4)), False)
    assert a[0]["model_id"] == b[0]["model_id"] != c[0]["model_id"]
