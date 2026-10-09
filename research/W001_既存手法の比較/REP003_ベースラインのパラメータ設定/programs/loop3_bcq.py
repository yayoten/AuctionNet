"""ループ 3：max_action を変えた BCQ の、生成モデルと最終の出力を測る（REP003.md 7.4 節）。

    .venv/bin/python <REP003>/programs/loop3_bcq.py <spec_name> [--threads 8]

上限（cap）は、保存した重みの vae.max_action から読む。測る量は 7.4 節の (i)〜(iv)。
出力：results/<spec_name>_bcq.csv、results/<spec_name>_alpha.npz（最終の出力 α[重み, 状態]）
"""
import argparse
import itertools
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
from probe_policy import MODELS, load_probe  # noqa: E402

RES = HERE / "results"
ap = argparse.ArgumentParser()
ap.add_argument("spec_name")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
states, logged, _, _, _ = load_probe(2000)

ws = []
for mp in sorted(MODELS.glob("M*/meta.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("rep") != "REP003" or m.get("spec_name") != a.spec_name or m["algo"] != "BCQ" or m.get("status") != "ok":
        continue
    ws.append((m["seed"], m["step_num"], mp.parent, m.get("train_kwargs") or {}))
    ws += [(m["seed"], json.loads(c.read_text(encoding="utf-8"))["step_num"], c.parent, m.get("train_kwargs") or {})
           for c in (mp.parent / "ckpt").glob("*/meta.json")]
ws.sort(key=lambda t: (t[0], t[1]))


def corr(u, v):
    return float(np.corrcoef(u, v)[0, 1]) if np.std(u) > 0 else float("nan")


rows, alphas = [], []
for seed, step, d, kw in ws:
    model = torch.jit.load(str(d / "bcq_model.pth"), map_location="cpu")
    with open(d / "normalize_dict.pkl", "rb") as f:
        nd = pickle.load(f)
    x = states.copy()
    for k, v in nd.items():
        x[:, k] = (x[:, k] - v["min"]) / (v["max"] - v["min"]) if v["max"] > v["min"] else 0.0
    x = torch.tensor(x, dtype=torch.float)
    vae, actor, critic = model.vae, model.actor, model.critic
    cap = float(vae.max_action)
    with torch.no_grad():
        torch.manual_seed(0)
        s = x.repeat_interleave(100, 0)
        z = torch.randn((s.shape[0], 2)).clamp(-0.5, 0.5)
        h1 = F.mish(vae.d1(torch.cat([s, z], 1)))
        pre = vae.d3(F.mish(vae.d2(h1.clamp(-0.5, 0.5))))                 # 学習と同じ経路（decode_withz）
        g_train = (cap * torch.tanh(pre)).reshape(len(x), 100)
        g_inf = cap * torch.tanh(vae.d3(F.mish(vae.d2(h1))))              # 推論の経路（decode）
        cand = actor(s, g_inf).reshape(len(x), 100)
        q = critic.q1(s, cand.reshape(-1, 1)).reshape(len(x), 100)
        chosen = cand.gather(1, q.argmax(1, keepdim=True)).reshape(-1)
        g_inf = g_inf.reshape(len(x), 100)
    al = chosen.numpy().astype(np.float64)
    alphas.append(al)
    rows.append(dict(seed=seed, step_num=step, max_action=cap, max_action_in_spec=kw.get("max_action"),
                     actor_max_action=float(actor.max_action),
                     g_train_share_ge99cap=float((g_train.abs() >= 0.99 * cap).float().mean()),          # (i)
                     g_train_share_le_minus99cap=float((g_train <= -0.99 * cap).float().mean()),
                     g_train_mean=float(g_train.mean()), g_train_std_over_states=float(g_train.mean(1).std()),
                     g_train_corr_to_logged=corr(g_train.mean(1).numpy(), logged),                       # (ii)
                     g_train_pre_tanh_median=float(pre.median()),
                     g_inf_share_ge99cap=float((g_inf.abs() >= 0.99 * cap).float().mean()),
                     alpha_mean=float(al.mean()), alpha_std_over_states=float(al.std()),                 # (iii)
                     alpha_share_ge99cap=float((np.abs(al) >= 0.99 * cap).mean()), alpha_share_le0=float((al <= 0).mean()),
                     alpha_corr_to_logged=corr(al, logged), alpha_rmse_to_logged=float(np.sqrt(((al - logged) ** 2).mean())),
                     cand_width_median=float((cand.max(1)[0] - cand.min(1)[0]).median()),                # (iv)
                     chosen_is_max_share=float((chosen >= cand.max(1)[0] - 1e-6).float().mean()),
                     chosen_minus_cand_mean=float((chosen - cand.mean(1)).mean())))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
t = pd.DataFrame(rows)
A = np.array(alphas)
# シード間の差（同じステップ）、ステップ間の差（最後の重みに対して）。定義は 4 節と同じ
for s_ in sorted(t.step_num.unique()):
    ix = t.index[t.step_num == s_].tolist()
    if len(ix) > 1:
        t.loc[ix, "seed_diff"] = np.mean([np.abs(A[i] - A[j]).mean() for i, j in itertools.combinations(ix, 2)])
last = t.step_num.max()
for sd in sorted(t.seed.unique()):
    il = t.index[(t.seed == sd) & (t.step_num == last)]
    if len(il):
        for i in t.index[t.seed == sd]:
            t.loc[i, "step_diff_to_last"] = np.abs(A[i] - A[il[0]]).mean()
t.to_csv(RES / f"{a.spec_name}_bcq.csv", index=False)
np.savez_compressed(RES / f"{a.spec_name}_alpha.npz", alpha=A, logged=logged, seed=t.seed.to_numpy(), step_num=t.step_num.to_numpy())
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(t.round(3).to_string())
