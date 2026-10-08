"""params.json から AuctionNet のシミュレーションを回し、結果を DB/runs/ に記録する。

    .venv/bin/python research/src/run_experiment.py <spec.json>   # Windows は .venv/Scripts/python.exe [--workers 4] [--force] [--dry-run]

spec.json（人が書く。変える値だけを書く。固定する値は書かない）
    {
      "name": "pid_default",                       # 人が読む名前（DB の spec_name に入る）
      "work": "W001", "rep": "REP001",             # どの作業・REPの実験か（任意）
      "note": "自由記述",
      "player": {"strategy": "PID", "kwargs": {"base_action": 15}},   # PID / ABid / OnlineLP / BC / IQL / CQL / BCQ / TD3_BC
                                                   # 学習ベースは kwargs.model_dir（リポジトリ直下からの相対パス。例 "DB/models/M..."）で
                                                   # 自前で学習した重みを指定する。書かなければ同梱の重み
      "player_indices": [0, 8, 16],                # プレイヤーを置く広告主の番号（0..47）
      "episodes": [0, 1],                          # エピソード番号（広告機会生成・環境ノイズの乱数シードになる）
      "seed": 1,                                   # numpy / torch の乱数シード
      "gin": {"PVNUM": 100000, "BiddingEnv.slot_coefficients": [1, 0.8, 0.6]},   # config/test.gin への上書き
      "sweep": {"player.kwargs.base_action": [5, 15, 30]}   # 任意。直積に展開して、それぞれ別の実行にする
    }

1実行（run）= 展開後の1つの設定 × プレイヤー位置1つ。run_id は「設定＋コードの版」から決まるので、
同じ設定をもう一度流しても同じ run_id になり、既にあればスキップする（--force で上書き）。
実行ごとに DB/runs/<run_id>/ に params.json / meta.json / episodes.parquet / ticks.parquet / agents.parquet を書く。
DuckDB への取り込みは `DB/build_db.py`。列の意味は `DB/columns.json`（DB/README.md）。
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GITHUB = REPO / "github"
DB_RUNS = REPO / "DB" / "runs"
# 戦略名 → (github.simul_bidding_env.strategy のモジュール名, クラス名)
STRATEGIES = {"PID": ("pid_bidding_strategy", "PidBiddingStrategy"), "ABid": ("abid_bidding_strategy", "AbidBiddingStrategy"),
              "OnlineLP": ("onlinelp_bidding_strategy", "OnlineLpBiddingStrategy"),
              "BC": ("bc_bidding_strategy", "BcBiddingStrategy"), "IQL": ("iql_bidding_strategy", "IqlBiddingStrategy"),
              "CQL": ("cql_bidding_strategy", "CqlBiddingStrategy"), "BCQ": ("bcq_bidding_strategy", "BcqBiddingStrategy"),
              "TD3_BC": ("td3_bc_bidding_strategy", "TD3_BCBiddingStrategy")}

# 評価は CPU で行う（REP001 と同じ条件。ロックの torch 1.12.0 は、新しい GPU では CUDA の計算が落ちる）。torch の import より前に隠す
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def strategy_class(name):
    import importlib
    mod, cls = STRATEGIES[name]
    return getattr(importlib.import_module(f"github.simul_bidding_env.strategy.{mod}"), cls)


def make_strategy(player):
    """spec の player から戦略を作る。model_dir はリポジトリ直下からの相対パスで書き、ここで絶対パスにする。"""
    kw = dict(player.get("kwargs", {}))
    if kw.get("model_dir"):
        kw["model_dir"] = str(REPO / kw["model_dir"])
    return strategy_class(player["strategy"])(**kw)


def model_info(player):
    """自前で学習した重みを使う run の、重みの出どころ（DB/models/<model_id>/meta.json から）。"""
    md = player.get("kwargs", {}).get("model_dir")
    if not md:
        return {"model_id": None, "model_dir": None}
    out = {"model_id": Path(md).name, "model_dir": md}
    mp = REPO / md / "meta.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        out.update(model_id=m.get("model_id", out["model_id"]), model_sha1=m.get("model_sha1"), model_step_num=m.get("step_num"),
                   model_spec_name=m.get("spec_name"))
    return out


# ----------------------------------------------------------------------------- 設定の展開
def _set_path(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand_spec(spec):
    """sweep を直積に展開して、sweep なしの設定のリストにする。各設定に sweep_point（振った値）を付ける。"""
    sweep = spec.get("sweep") or {}
    base = {k: v for k, v in spec.items() if k != "sweep"}
    if not sweep:
        return [dict(base, sweep_point={})]
    keys = list(sweep)
    out = []
    for values in itertools.product(*[sweep[k] for k in keys]):
        s = copy.deepcopy(base)
        for k, v in zip(keys, values):
            _set_path(s, k, v)
        s["sweep_point"] = dict(zip(keys, values))
        out.append(s)
    return out


def to_gin_value(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "None"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(to_gin_value(x) for x in v) + "]"
    if isinstance(v, str):
        return json.dumps(v)
    return repr(v)


def gin_bindings(gin_dict):
    return [f"{k} = {to_gin_value(v)}" for k, v in (gin_dict or {}).items()]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------------------- コードの版
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout.strip()
    except Exception:
        return ""


def code_version():
    """github/ の中身の版。コミット済みなら tree のハッシュ、未コミットの変更があれば差分のハッシュも足す。"""
    tree = _git("rev-parse", "HEAD:github")
    diff = _git("diff", "HEAD", "--", "github")
    untracked = _git("ls-files", "--others", "--exclude-standard", "github")
    dirty = bool(diff or untracked)
    ver = tree + (("+" + hashlib.sha1((diff + untracked).encode("utf-8", "replace")).hexdigest()[:8]) if dirty else "")
    return {"github_tree": tree, "github_dirty": dirty, "github_version": ver, "head_commit": _git("rev-parse", "HEAD"),
            "repo_dirty_paths": [l for l in _git("status", "--porcelain").splitlines()][:30]}


def make_run_id(resolved, player_index, version):
    key = {k: resolved.get(k) for k in ("player", "episodes", "seed", "gin")}
    h = hashlib.sha1((canonical(key) + "|" + version).encode("utf-8")).hexdigest()[:10]
    return f"R{h}-p{player_index:02d}"


# ----------------------------------------------------------------------------- 1実行
class TimedStrategy:
    """プレイヤー戦略の bidding 1回の所要時間を測るための透過プロキシ。"""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "last_bidding_seconds", 0.0)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, key, value):
        setattr(self._inner, key, value)

    def bidding(self, *args, **kwargs):
        t = time.perf_counter()
        r = self._inner.bidding(*args, **kwargs)
        object.__setattr__(self, "last_bidding_seconds", time.perf_counter() - t)
        return r


def _setup_imports():
    for p in (str(REPO), str(GITHUB / "strategy_train_env")):
        if p not in sys.path:
            sys.path.append(p)
    os.chdir(GITHUB)


def gin_value(name):
    """gin で束ねた値。%PVNUM のような macro は値に解決する。"""
    import gin
    v = gin.query_parameter(name)
    if type(v).__name__ == "ConfigurableReference":
        v = gin.query_parameter(str(v))
    return v


def _defaults_of(callable_):
    import inspect
    out = {}
    for n, p in inspect.signature(callable_).parameters.items():
        if n in ("self", "args", "kwargs") or p.default is inspect.Parameter.empty:
            continue
        v = p.default
        if hasattr(v, "tolist"):
            v = v.tolist()
        out[n] = v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else repr(v)
    return out


def effective_params(resolved):
    """spec に書かれていない値も含めた、実際に使われた値（コード側の既定値を補ったもの）。"""
    import gin
    from github.simul_bidding_env.Controller.Controller import Controller
    from github.simul_bidding_env.Environment.BiddingEnv import BiddingEnv
    from github.simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis
    s = resolved["player"]["strategy"]
    kw = _defaults_of(strategy_class(s).__init__)
    kw.update(resolved["player"].get("kwargs", {}))
    # gin で束ねた値（macro 解決後）。束ねていないものは各クラスの既定値
    eff_env = {}
    for name, klass in (("Controller", Controller), ("BiddingEnv", BiddingEnv), ("PlayerAnalysis", PlayerAnalysis)):
        d = _defaults_of(klass.__init__)
        for k in list(d):
            try:
                v = gin_value(f"{name}.{k}")
                d[k] = v.tolist() if hasattr(v, "tolist") else v
            except Exception:
                pass
        eff_env[name] = d
    return {"player": {"strategy": s, "kwargs": kw}, "env": eff_env, "episodes": resolved["episodes"],
            "seed": resolved["seed"], "gin_bindings": gin_bindings(resolved.get("gin"))}


def execute_run(task):
    """1つの run を実行して、DB/runs/<run_id>/ に書く。meta の dict を返す。"""
    resolved, player_index, run_id, version = task["resolved"], task["player_index"], task["run_id"], task["version"]
    out_dir = DB_RUNS / run_id
    tmp_dir = DB_RUNS / (run_id + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    meta = {"run_id": run_id, "spec_name": resolved.get("name"), "work": resolved.get("work"), "rep": resolved.get("rep"),
            "note": resolved.get("note"), "sweep_point": resolved.get("sweep_point", {}), "player_index": player_index,
            "strategy": resolved["player"]["strategy"], "created_at": datetime.now().isoformat(timespec="seconds"),
            **{k: version[k] for k in ("head_commit", "github_tree", "github_dirty", "github_version", "repo_dirty_paths")},
            "host": platform.node(), "os": platform.platform(), "cpu_count": os.cpu_count(), "status": "running"}
    t0 = time.time()
    try:
        _setup_imports()
        import gin
        import numpy as np
        import pandas as pd
        import psutil
        import torch
        from github.run.run_test import run_test
        torch.set_num_threads(1)
        meta.update(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    pandas=pd.__version__)

        gin.clear_config()
        gin.parse_config_files_and_bindings([str(GITHUB / "config" / "test.gin")], gin_bindings(resolved.get("gin")))
        seed = resolved["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)

        eff = effective_params(resolved)
        (tmp_dir / "params.json").write_text(json.dumps({"spec": resolved, "effective": eff}, ensure_ascii=False,
                                                        indent=2), encoding="utf-8")
        meta.update(model_info(resolved["player"]))
        strategy = TimedStrategy(make_strategy(resolved["player"]))

        min_remaining_budget = eff["env"]["BiddingEnv"]["min_remaining_budget"]
        beta = eff["env"]["PlayerAnalysis"]["penalty_beta"]
        p = player_index
        tick_rows, agent_rows = [], []
        ep_state = {}
        proc = psutil.Process(os.getpid())
        peak = [0.0]
        last_t = [time.perf_counter()]

        def hook(info):
            ep, tick, n = info["episode"], info["tick"], info["num_pv"]
            now = time.perf_counter()
            wall = now - last_t[0]
            bids, pv, slot, cost, reward = info["bids"], info["pv_values"], info["slot"], info["cost"], info["reward"]
            agents = info["agents"]
            bp, pp = bids[:, p], pv[:, p]
            alpha_all = bids.sum(axis=0) / (pv.sum(axis=0) + 1e-12)       # 全広告主の実効入札係数（入札額の和/価値の和）
            sp = slot[p]
            st = ep_state.setdefault(ep, {"reward": np.zeros(len(agents)), "cost": np.zeros(len(agents))})
            st["reward"] += reward
            st["cost"] += cost
            rss = proc.memory_info().rss / 2 ** 20
            peak[0] = max(peak[0], rss)
            tick_rows.append(dict(
                episode=ep, tick=tick, num_pv=n, pvalue_mean=float(pp.mean()), pvalue_std=float(pp.std()),
                pvalue_sigma_mean=float(info["pvalue_sigmas"][:, p].mean()),
                bid_mean=float(bp.mean()), bid_nonzero_ratio=float((bp > 0).mean()),
                alpha_eff=float(alpha_all[p]), alpha_rank=int((alpha_all > alpha_all[p]).sum() + 1),
                alpha_others_mean=float(np.delete(alpha_all, p).mean()),
                alpha_others_median=float(np.median(np.delete(alpha_all, p))),
                n_won=int((sp > 0).sum()), n_slot1=int((sp == 1).sum()), n_slot2=int((sp == 2).sum()),
                n_slot3=int((sp == 3).sum()), n_exposed=int(info["is_exposed"][p].sum()),
                reward=float(reward[p]), cost=float(cost[p]),
                remaining_budget_before=float(info["remaining_budget_before"][p]),
                lwc_mean=float(np.mean(info["least_winning_cost"])),
                total_cost_all=float(cost.sum()),
                n_active_agents=int((info["remaining_budget_before"] >= min_remaining_budget).sum()),
                bidding_seconds=float(strategy.last_bidding_seconds), tick_wall_seconds=float(wall),
                rss_mb=float(rss)))
            if tick == info["num_tick_total"] - 1:
                for i, a in enumerate(agents):
                    agent_rows.append(dict(episode=ep, agent_index=i, agent_name=a.name, category=int(a.category),
                                           budget=float(a.budget), cpa_constraint=float(a.cpa),
                                           reward=float(st["reward"][i]), cost=float(st["cost"][i]),
                                           final_remaining_budget=float(a.remaining_budget)))
            last_t[0] = time.perf_counter()

        # tick_hook に num_tick_total と sigma を渡すため、薄く包む
        num_tick = gin_value("run_test.num_tick")

        def hook2(info):
            info["num_tick_total"] = num_tick
            hook(info)

        t_sim = time.time()
        res = run_test(player_index=p, player_agent=strategy, episode_ids=resolved["episodes"], tick_hook=hook2)
        sim_seconds = time.time() - t_sim

        # --- 集計 ---
        raw = pd.DataFrame(res["rawData"])
        eps, agent_frames = [], []
        agents_df = pd.DataFrame(agent_rows)
        for _, r in raw.iterrows():
            ep = int(r["episode"])
            a = agents_df[agents_df.episode == ep].copy()
            cpa_real = a.cost / (a.reward + 1e-10)
            pen = np.where(cpa_real > a.cpa_constraint, (a.cpa_constraint / (cpa_real + 1e-10)) ** beta, 1.0)
            a["real_cpa"] = cpa_real
            a["penalty"] = pen
            a["score_component"] = pen * a.reward
            a["score_rank"] = a.score_component.rank(ascending=False, method="min").astype(int)
            a["is_player"] = a.agent_index == p
            agent_frames.append(a)
            mine = a[a.is_player].iloc[0]
            eps.append(dict(
                episode=ep, category=int(res["category"]), cpa_constraint=float(r["cpaConstraint"]),
                budget=float(r["budget"]), reward=float(r["reward"]), all_cost=float(r["allCost"]),
                real_cpa=float(r["cpa"]), penalty=float(mine["penalty"]), score_component=float(r["score"]),
                score_rank=int(mine["score_rank"]), n_agents=int(len(a)),
                all_compete_pv=float(r["allCompetePv"]), all_win_pv=float(r["allWinPv"]),
                win_pv_ratio=float(r["win_pv_ratio"]), budget_consumer_ratio=float(r["budget_consumer_ratio"]),
                second_price_ratio=float(r["second_price_ratio"]),
                cpa_exceedance_rate=float(r["cpa_exceedance_Rate"]),
                last_compete_tick_index=int(r["last_compete_tick_index"]), bid_mean=float(r["bidMean"]),
                others_score_mean=float(a[~a.is_player].score_component.mean())))
        pd.DataFrame(eps).assign(run_id=run_id).to_parquet(tmp_dir / "episodes.parquet", index=False)
        pd.DataFrame(tick_rows).assign(run_id=run_id).to_parquet(tmp_dir / "ticks.parquet", index=False)
        pd.concat(agent_frames).assign(run_id=run_id).to_parquet(tmp_dir / "agents.parquet", index=False)
        meta.update(status="ok", score=float(res["score"]), reward_total=float(res["reward"]),
                    sim_seconds=sim_seconds, peak_rss_mb=peak[0],
                    bidding_seconds_total=float(sum(t["bidding_seconds"] for t in tick_rows)),
                    bidding_seconds_per_call_mean=float(np.mean([t["bidding_seconds"] for t in tick_rows])),
                    n_episodes=len(resolved["episodes"]), n_ticks=num_tick)
    except Exception as e:  # 止めずに記録して先へ
        meta.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    meta["wall_seconds"] = time.time() - t0
    (tmp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return meta


# ----------------------------------------------------------------------------- 入口
def strategy_defaults(strategy):
    """戦略のコンストラクタの既定値（github を import して調べる。重い依存は読まない）。"""
    _setup_imports_light()
    return _defaults_of(strategy_class(strategy).__init__)


def _setup_imports_light():
    for p in (str(REPO), str(GITHUB / "strategy_train_env")):
        if p not in sys.path:
            sys.path.append(p)


def build_tasks(spec_path, force):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    version = code_version()
    tasks, skipped = [], []
    for resolved in expand_spec(spec):
        resolved.setdefault("episodes", [0, 1])
        resolved.setdefault("seed", 1)
        resolved.setdefault("player_indices", [0, 1])
        assert resolved["player"]["strategy"] in STRATEGIES, f"strategy は {list(STRATEGIES)} のどれか"
        # 既定値と同じ指定は取り除く（同じ実効条件なら同じ run_id にして、既にある結果を使い回すため）
        defaults = strategy_defaults(resolved["player"]["strategy"])
        kw = resolved["player"].get("kwargs", {})
        resolved["player"]["kwargs"] = {k: v for k, v in kw.items() if not (k in defaults and defaults[k] == v)}
        resolved["spec_file"] = str(Path(spec_path).as_posix())
        for idx in resolved["player_indices"]:
            rid = make_run_id(resolved, idx, version["github_version"])
            mp = DB_RUNS / rid / "meta.json"
            if mp.exists() and not force and json.loads(mp.read_text(encoding="utf-8")).get("status") == "ok":
                skipped.append(rid)  # status=ok の run だけスキップ。失敗した run は流し直す
                continue
            tasks.append(dict(resolved=resolved, player_index=idx, run_id=rid, version=version))
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="既にある run を上書きする")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    DB_RUNS.mkdir(parents=True, exist_ok=True)
    tasks, skipped = build_tasks(args.spec, args.force)
    print(f"実行 {len(tasks)} 件 / 既存でスキップ {len(skipped)} 件 / workers={args.workers}", flush=True)
    if args.dry_run:
        for t in tasks:
            print(t["run_id"], t["resolved"]["player"], t["resolved"].get("sweep_point"))
        return
    t0 = time.time()
    done = 0
    if args.workers <= 1:
        results = map(execute_run, tasks)
        pool = None
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.workers)
        results = pool.imap_unordered(execute_run, tasks)
    for m in results:
        done += 1
        info = (f"score={m.get('score', float('nan')):.5f} sim={m.get('sim_seconds', 0):.0f}s"
                if m["status"] == "ok" else m.get("error"))
        print(f"[{done}/{len(tasks)}] {m['run_id']} {m['status']} {m['strategy']} p{m['player_index']} "
              f"{m['sweep_point']} {info}", flush=True)
    if pool:
        pool.close()
        pool.join()
    print(f"完了 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
