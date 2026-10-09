"""ループ 2：ループ 1 の重みの中身を測り、出力を決めている経路を切り分ける（REP003.md 7.2 節。新しい学習はしない）。

    .venv/bin/python <REP003>/programs/loop2_diagnose.py [--threads 8]

2a BCQ    ：生成モデルの出力を、学習と同じ経路（decode_withz）と、推論の経路（decode）で比べる。候補 100 個の幅も出す。
2b TD3_BC ：方策の損失の 2 つの項（Q の項、二乗誤差の項）の、方策の出力についての傾きの大きさの比 r。
2c CQL    ：a を 0〜300 で振ったときの min(Q1, Q2) の最大の位置 a* と、方策の出力の距離。
出力：results/loop2_bcq.csv、loop2_td3bc.csv、loop2_cql.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "research" / "src"))
from probe_policy import FILES, MODELS, load_probe  # noqa: E402

RES = HERE / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--spec", default="loop1_train")
ap.add_argument("--tag", default="loop2")
a = ap.parse_args()
torch.set_num_threads(a.threads)
states, logged, _, _, _ = load_probe(2000)
logged_t = torch.tensor(logged, dtype=torch.float).reshape(-1, 1)


def weights(algo, steps):
    """(seed, step, フォルダ) を、最後の重みとチェックポイントから拾う。"""
    out = []
    for mp in sorted(MODELS.glob("M*/meta.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("rep") != "REP003" or m.get("spec_name") != a.spec or m["algo"] != algo or m.get("status") != "ok":
            continue
        cands = [(m["step_num"], mp.parent)] + [(json.loads(c.read_text(encoding="utf-8"))["step_num"], c.parent)
                                                for c in (mp.parent / "ckpt").glob("*/meta.json")]
        out += [(m["seed"], s, d) for s, d in cands if steps is None or s in steps]
    return sorted(out, key=lambda t: (t[0], t[1]))


def load(d, algo):
    model = torch.jit.load(str(d / FILES[algo]), map_location="cpu")
    with open(d / "normalize_dict.pkl", "rb") as f:
        nd = pickle.load(f)
    x = states.copy()
    for k, v in nd.items():
        x[:, k] = (x[:, k] - v["min"]) / (v["max"] - v["min"]) if v["max"] > v["min"] else 0.0
    return model, torch.tensor(x, dtype=torch.float)


def corr(u, v):
    return float(np.corrcoef(u, v)[0, 1]) if np.std(u) > 0 else float("nan")


# ---- 2a BCQ ----
rows = []
for seed, step, d in weights("BCQ", None):
    model, x = load(d, "BCQ")
    vae, actor, critic = model.vae, model.actor, model.critic
    with torch.no_grad():
        torch.manual_seed(0)
        s = x.repeat_interleave(100, 0)                                   # 状態ごとに候補 100 個
        z = torch.randn((s.shape[0], 2)).clamp(-0.5, 0.5)
        h1 = F.mish(vae.d1(torch.cat([s, z], 1)))                         # 1 層目の出力（推論の経路では切らない）
        g_inf = 100 * torch.tanh(vae.d3(F.mish(vae.d2(h1))))              # decode と同じ計算
        pre_inf = vae.d3(F.mish(vae.d2(h1)))
        g_train = 100 * torch.tanh(vae.d3(F.mish(vae.d2(h1.clamp(-0.5, 0.5)))))   # decode_withz と同じ計算
        check = float((g_inf - 100 * torch.tanh(vae.d3(F.mish(vae.d2(F.mish(vae.d1(torch.cat([s, z], 1)))))))).abs().max())
        cand = actor(s, g_inf).reshape(len(x), 100)
        q = critic.q1(s, cand.reshape(-1, 1)).reshape(len(x), 100)
        chosen = cand.gather(1, q.argmax(1, keepdim=True)).reshape(-1)
        gi, gt = g_inf.reshape(len(x), 100), g_train.reshape(len(x), 100)
    rows.append(dict(seed=seed, step_num=step,
                     g_train_share_ge99=float((gt.abs() >= 99).float().mean()), g_train_mean=float(gt.mean()),
                     g_train_std_over_states=float(gt.mean(1).std()), g_train_corr_to_logged=corr(gt.mean(1).numpy(), logged),
                     g_inf_share_ge99=float((gi.abs() >= 99).float().mean()), g_inf_mean=float(gi.mean()),
                     g_inf_pre_tanh_median=float(pre_inf.median()),
                     h1_share_outside_half=float((h1.abs() > 0.5).float().mean()),
                     cand_width_median=float((cand.max(1)[0] - cand.min(1)[0]).median()),
                     chosen_mean=float(chosen.mean()), chosen_is_max_share=float((chosen >= cand.max(1)[0] - 1e-6).float().mean()),
                     recompute_check=check))
    print("BCQ", rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / f"{a.tag}_bcq.csv", index=False)

# ---- 2b TD3_BC ----
rows = []
for seed, step, d in weights("TD3_BC", {10000, 20000, 30000, 50000}):
    model, x = load(d, "TD3_BC")
    c = model.critic
    pi = model.actor(x).detach().requires_grad_(True)
    Q = c.l3(F.relu(c.l2(F.relu(c.l1(torch.cat([x, pi], 1))))))           # Critic.Q1 と同じ計算（101 個のロジット）
    lmbda = 2.5 / Q.abs().mean().detach()
    term_q = -lmbda * Q.mean()
    term_bc = F.mse_loss(pi, logged_t)
    gq, = torch.autograd.grad(term_q, pi, retain_graph=True)
    gb, = torch.autograd.grad(term_bc, pi)
    rows.append(dict(seed=seed, step_num=step, term_q=float(term_q), term_bc=float(term_bc),
                     grad_q_abs_median=float(gq.abs().median()), grad_bc_abs_median=float(gb.abs().median()),
                     r=float(gq.abs().median() / gb.abs().median()), r_mean=float(gq.abs().mean() / gb.abs().mean()),
                     q_logits_dim=int(Q.shape[1])))
    print("TD3_BC", rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / f"{a.tag}_td3bc.csv", index=False)

# ---- 2c CQL ----
rows = []
grid = torch.arange(0, 301, dtype=torch.float)
for seed, step, d in weights("CQL", {10000, 20000, 30000, 50000}):
    model, x = load(d, "CQL")
    with torch.no_grad():
        pi = model(x).reshape(-1)
        astar, qgap = [], []
        for i in range(0, len(x), 200):
            xs = x[i:i + 200]
            sa = torch.cat([xs.repeat_interleave(len(grid), 0), grid.repeat(len(xs)).reshape(-1, 1)], 1)
            q = torch.min(model.qf1(sa), model.qf2(sa)).reshape(len(xs), len(grid))
            astar.append(grid[q.argmax(1)])
            qgap.append(q.max(1)[0] - q.min(1)[0])
        astar, qgap = torch.cat(astar), torch.cat(qgap)
    rows.append(dict(seed=seed, step_num=step, pi_mean=float(pi.mean()), astar_mean=float(astar.mean()),
                     astar_at_edge_share=float(((astar == 0) | (astar == 300)).float().mean()),
                     astar_at_0_share=float((astar == 0).float().mean()), astar_at_300_share=float((astar == 300).float().mean()),
                     dist_pi_astar_median=float((pi - astar).abs().median()),
                     corr_pi_astar=corr(pi.numpy(), astar.numpy()), q_range_over_grid_median=float(qgap.median()),
                     policy_std=float(model.policy.log_std.exp())))
    print("CQL", rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / f"{a.tag}_cql.csv", index=False)
