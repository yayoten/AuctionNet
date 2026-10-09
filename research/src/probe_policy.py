"""学習した重みが、決まった状態の集まり（プローブ）に対して出す α を測る。

    .venv/bin/python research/src/probe_policy.py --rep REP003 --out <出力フォルダ> [--n 2000] [--threads 8]

シミュレータを回さずに、「学習がどこまで進んだか」「乱数シードで方策が変わるか」を、方策の出力そのもので見るための道具。
- プローブは、学習データ（rlData）から固定の乱数で抜いた n 個の状態。全モデルに同じものを使う。
- 状態の正規化は、シミュレータ側の戦略（*_bidding_strategy.py）と同じ式（(x−min)/(max−min)）で行う。
- 対象は、DB/models/ のうち meta.rep が --rep のモデルと、そのチェックポイント（ckpt/）。
出力：probe_index.csv（1 行 = 重み 1 つ。α の平均・中央値・0 以下の割合・ログの α との誤差など）、
      probe_alpha.npz（alpha[重み, 状態]、ログの α、状態の添字）。
"""
import argparse
import ast
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "DB" / "models"
DATA = REPO / "DB" / "dataset" / "traffic" / "training_data_rlData_folder" / "training_data_all-rlData.csv"
FILES = {"BC": "bc_model.pth", "IQL": "iql_model.pth", "CQL": "cql_model.pth", "BCQ": "bcq_model.pth", "TD3_BC": "td3_bc_model.pth"}


def load_probe(n=2000, seed=0, data=DATA):
    """プローブの状態（生の 16 次元）、ログの α、ティック、行の添字。"""
    rl = pd.read_csv(data, usecols=["state", "action", "timeStepIndex", "deliveryPeriodIndex", "advertiserNumber"])
    idx = np.sort(np.random.default_rng(seed).choice(len(rl), size=min(n, len(rl)), replace=False))
    sub = rl.iloc[idx]
    states = np.array([ast.literal_eval(s) for s in sub.state], dtype=np.float64)
    return states, sub.action.to_numpy(), sub.timeStepIndex.to_numpy().astype(int), idx, rl.action.to_numpy()


def alpha_of(model_dir, algo, states):
    """重み 1 つが、各状態に出す α（戦略と同じ正規化。0 での切り捨てなどは、モデルの forward に入っているものだけ）。"""
    model = torch.jit.load(str(Path(model_dir) / FILES[algo]), map_location="cpu")
    with open(Path(model_dir) / "normalize_dict.pkl", "rb") as f:
        nd = pickle.load(f)
    x = states.copy()
    for k, v in nd.items():
        x[:, k] = (x[:, k] - v["min"]) / (v["max"] - v["min"]) if v["max"] > v["min"] else 0.0
    x = torch.tensor(x, dtype=torch.float)
    torch.manual_seed(0)                       # BCQ は推論で乱数を引く
    with torch.no_grad():
        if algo == "BCQ":                      # 1 状態ずつ（中で 100 個の候補を作る）。同梱の重みは eval_flag を取る
            takes_flag = "eval_flag" in str(model.forward.schema)
            out = [float(model(s, eval_flag=True) if takes_flag else model(s)) for s in x]
        else:
            out = model(x).reshape(-1).tolist()
    return np.asarray(out, dtype=np.float64)


def model_dirs(rep):
    """(meta, フォルダ) を、最後の重みとチェックポイントの両方について返す。"""
    out = []
    for mp in sorted(MODELS.glob("M*/meta.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("rep") != rep or m.get("status") != "ok":
            continue
        out.append((m, mp.parent))
        for cp in sorted((mp.parent / "ckpt").glob("*/meta.json")):
            c = json.loads(cp.read_text(encoding="utf-8"))
            c["spec_name"] = m.get("spec_name")
            out.append((c, cp.parent))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    states, logged, ticks, idx, all_actions = load_probe(a.n)
    p99 = float(np.percentile(all_actions, 99))
    old = {}
    if (out / "probe_alpha.npz").exists() and (out / "probe_index.csv").exists():      # 測り直さない（重みの SHA-1 で照合）
        z = np.load(out / "probe_alpha.npz")
        if np.array_equal(z["state_index"], idx):
            old = dict(zip(pd.read_csv(out / "probe_index.csv").weights_sha1, z["alpha"]))
    rows, alphas = [], []
    for m, d in model_dirs(a.rep):
        w = m.get("weights_sha1")
        al = old[w] if w in old else alpha_of(d, m["algo"], states)
        alphas.append(al)
        rows.append(dict(
            model_id=m["model_id"], parent_model_id=m.get("parent_model_id", m["model_id"]), spec_name=m.get("spec_name"),
            algo=m["algo"], seed=m["seed"], step_num=m["step_num"], is_checkpoint=bool(m.get("is_checkpoint", False)),
            weights_sha1=w, alpha_mean=al.mean(), alpha_median=np.median(al), alpha_std=al.std(),
            share_le0=float((al <= 0).mean()), share_ge99=float((np.abs(al) >= 99).mean()),
            share_in_logged_range=float(((al >= 0) & (al <= p99)).mean()),
            rmse_to_logged=float(np.sqrt(((al - logged) ** 2).mean())),
            corr_to_logged=float(np.corrcoef(al, logged)[0, 1]) if al.std() > 0 else float("nan")))
        print(rows[-1]["algo"], rows[-1]["seed"], rows[-1]["step_num"], round(rows[-1]["alpha_mean"], 2), flush=True)
    pd.DataFrame(rows).to_csv(out / "probe_index.csv", index=False)
    np.savez_compressed(out / "probe_alpha.npz", alpha=np.array(alphas), logged=logged, tick=ticks, state_index=idx)
    print(f"プローブ {len(idx)} 状態 × 重み {len(rows)} 個 → {out}")


if __name__ == "__main__":
    main()
