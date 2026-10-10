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
      "sweep": {"player.kwargs.base_action": [5, 15, 30]},  # 任意。直積に展開して、それぞれ別の実行にする
      "record": "standard"                         # 記録の段（任意。既定は standard）。basic / standard / detail / full。
                                                   # run_id には入らない（結果は同じで、残す量だけが違う）。DB/記録の一覧.md
    }

1実行（run）= 展開後の1つの設定 × プレイヤー位置1つ。run_id は「設定＋コードの版」から決まるので、
同じ設定をもう一度流しても同じ run_id になり、既にあればスキップする（--force で上書き）。
実行ごとに DB/runs/<run_id>/ に params.json / meta.json / episodes.parquet / ticks.parquet / agents.parquet を書く。
sim_iters.parquet は常に書く。記録の段が standard 以上なら raw/agent_ticks.parquet と raw/pv_won.parquet、detail 以上なら raw/pv_all.parquet、
full なら raw/bids_all_ep*.parquet も書く（raw/ は git 管理外）。同じ run_id が、低い段で既にあるときは、流し直して置き換える。
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
# 記録の段（数が大きいほど、残す量が多い）。何を残すかは DB/記録の一覧.md
RECORD_LEVELS = {"basic": 0, "standard": 1, "detail": 2, "full": 3}
RECORD_VERSION = 2          # 記録の形式の版。1 = 2026-10-09 まで（episodes / ticks / agents だけ）。2 = T006 で拡充
PARQUET_KW = dict(index=False, compression="zstd")
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


def _github_tree_without_tests():
    """HEAD の github/ から Test/ を除いた tree のハッシュ。Test/ が無いときは `git rev-parse HEAD:github` と同じ値になる。"""
    try:
        ls = subprocess.run(["git", "ls-tree", "-z", "HEAD:github"], cwd=REPO, capture_output=True, timeout=60).stdout
        kept = b"".join(e + b"\0" for e in ls.split(b"\0") if e and e.split(b"\t", 1)[1] != b"Test")
        return subprocess.run(["git", "mktree", "-z"], cwd=REPO, input=kept, capture_output=True, timeout=60).stdout.decode().strip()
    except Exception:
        return ""


def code_version():
    """github/ の中身の版。コミット済みなら tree のハッシュ、未コミットの変更があれば差分のハッシュも足す。

    github/Test/（pytest）は数えない。テストを足したり直したりしても、run_id・model_id は変わらない。
    """
    no_tests = ":(exclude)github/Test"
    tree = _github_tree_without_tests()
    diff = _git("diff", "HEAD", "--", "github", no_tests)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "github", no_tests)
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


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _lib_versions():
    out = {}
    for name in ("scipy", "gin", "psutil", "pyarrow", "duckdb"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", None)
            if out[name] is None:
                from importlib import metadata
                out[name] = metadata.version("gin-config" if name == "gin" else name)
        except Exception:
            out[name] = None
    return out


def _q(x, qs=(10, 50, 90)):
    """列ごとの分位点。x は (機会, 広告主)。"""
    import numpy as np
    return np.percentile(x, qs, axis=0) if x.shape[0] else np.zeros((len(qs), x.shape[1]))


def execute_run(task):
    """1つの run を実行して、DB/runs/<run_id>/ に書く。meta の dict を返す。"""
    resolved, player_index, run_id, version = task["resolved"], task["player_index"], task["run_id"], task["version"]
    level_name = resolved.get("record", "standard")
    level = RECORD_LEVELS[level_name]
    runs_dir = Path(task.get("runs_dir") or DB_RUNS)
    out_dir = runs_dir / run_id
    tmp_dir = runs_dir / (run_id + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    import resource
    t0 = time.time()
    cpu0 = time.process_time()
    try:
        load0 = os.getloadavg()
    except Exception:
        load0 = (None, None, None)
    meta = {"run_id": run_id, "spec_name": resolved.get("name"), "work": resolved.get("work"), "rep": resolved.get("rep"),
            "note": resolved.get("note"), "sweep_point": resolved.get("sweep_point", {}), "player_index": player_index,
            "strategy": resolved["player"]["strategy"], "created_at": datetime.now().isoformat(timespec="seconds"),
            **{k: version[k] for k in ("head_commit", "github_tree", "github_dirty", "github_version", "repo_dirty_paths")},
            "host": platform.node(), "os": platform.platform(), "cpu_count": os.cpu_count(), "status": "running",
            # ここから下は、記録の形式の版 2（T006）で足したもの
            "record_level": level_name, "record_level_num": level, "record_version": RECORD_VERSION,
            "started_at_epoch": t0, "pid": os.getpid(), "n_workers": task.get("n_workers"),
            "n_tasks_in_batch": task.get("n_tasks"), "cpu_model": _cpu_model(), "machine": platform.machine(),
            "python_executable": sys.executable, "argv": " ".join(sys.argv),
            "loadavg1_start": load0[0], "loadavg5_start": load0[1],
            "env_threads": json.dumps({k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")}),
            "run_experiment_sha1": _sha1_file(__file__)}
    try:
        t_imp = time.time()
        _setup_imports()
        import gin
        import numpy as np
        import pandas as pd
        import psutil
        import torch
        from github.run.run_test import run_test
        torch.set_num_threads(1)
        import_seconds = time.time() - t_imp
        vm = psutil.virtual_memory()
        meta.update(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                    pandas=pd.__version__, lib_versions=json.dumps(_lib_versions()),
                    torch_num_threads=torch.get_num_threads(), import_seconds=import_seconds,
                    mem_total_mb=vm.total / 2 ** 20, mem_available_start_mb=vm.available / 2 ** 20)

        gin.clear_config()
        gin_file = GITHUB / "config" / "test.gin"
        gin.parse_config_files_and_bindings([str(gin_file)], gin_bindings(resolved.get("gin")))
        seed = resolved["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)

        eff = effective_params(resolved)
        (tmp_dir / "params.json").write_text(json.dumps({"spec": resolved, "effective": eff}, ensure_ascii=False,
                                                        indent=2), encoding="utf-8")
        sf = REPO / str(resolved.get("spec_file", ""))
        meta.update(model_info(resolved["player"]))
        meta.update(gin_file_sha1=_sha1_file(gin_file), gin_config=gin.config_str(),
                    spec_file_sha1=_sha1_file(sf) if sf.is_file() else None,
                    seeds=json.dumps(dict(
                        numpy_torch=seed, episodes=list(resolved["episodes"]),
                        env_default_seed=eff["env"]["BiddingEnv"]["default_seed"],
                        env_conversion_seed=eff["env"]["BiddingEnv"]["conversion_seed"],
                        adjust_over_cost_seed=1, pvgen_module_seed=1019, trunc_magic_number=1019,
                        note="露出・雑音・抽選の乱数は、呼ぶたびに同じ種で作り直される。広告機会はエピソード番号が種")))
        t_strat = time.time()
        strategy = TimedStrategy(make_strategy(resolved["player"]))
        strategy_init_seconds = time.time() - t_strat

        min_remaining_budget = eff["env"]["BiddingEnv"]["min_remaining_budget"]
        reserve = eff["env"]["BiddingEnv"]["reserve_pv_price"]
        beta = eff["env"]["PlayerAnalysis"]["penalty_beta"]
        p = player_index
        tick_rows, agent_rows, agent_tick_frames, iter_rows, pv_won_frames, pv_all_frames = [], [], [], [], [], []
        bids_all = {}            # full のとき：エピソード → [ティックごとの表]
        ep_state = {}
        proc = psutil.Process(os.getpid())
        peak = [0.0]
        last_t = [time.perf_counter()]
        last_cpu = [time.process_time()]
        f32 = np.float32

        def pv_frame(info, rows):
            """プレイヤーの機会ごとの記録（rows：その機会の番号）。"""
            ep, tick = info["episode"], info["tick"]
            bids, pv, sig, vr = info["bids"], info["pv_values"], info["pvalue_sigmas"], info["values_real"]
            sb = -np.sort(-bids[rows], axis=1)[:, :4] if len(rows) else np.zeros((0, 4))
            w = info["winner"][rows]
            n = len(rows)
            return pd.DataFrame(dict(
                episode=np.full(n, ep, np.int16), tick=np.full(n, tick, np.int8), pv_index=rows.astype(np.int32),
                pvalue=pv[rows, p].astype(f32), sigma=sig[rows, p].astype(f32), value_real=vr[rows, p].astype(f32),
                bid=bids[rows, p].astype(f32), bid_before_adjust=info["bids_before_adjust"][rows, p].astype(f32),
                slot=info["slot"][p][rows].astype(np.int8), price=info["cost_pit"][p][rows].astype(f32),
                exposure_draw=info["exposure_draw"][rows, p].astype(np.int8),
                is_exposed=info["is_exposed"][p][rows].astype(np.int8),
                conversion_draw=info["conversion_draw"][rows, p].astype(np.int8),
                conversion=info["conversion"][p][rows].astype(np.int8),
                bid_rank=((bids[rows] > bids[rows, p][:, None]).sum(axis=1) + 1).astype(np.int8),
                top1_bid=sb[:, 0].astype(f32), top2_bid=sb[:, 1].astype(f32), top3_bid=sb[:, 2].astype(f32),
                top4_bid=sb[:, 3].astype(f32),
                winner1=w[:, 0].astype(np.int8), winner2=w[:, 1].astype(np.int8), winner3=w[:, 2].astype(np.int8)))

        def hook(info):
            t_hook = time.perf_counter()
            ep, tick, n = info["episode"], info["tick"], info["num_pv"]
            now = time.perf_counter()
            wall = now - last_t[0]
            cpu_now = time.process_time()
            bids, pv, slot, cost, reward = info["bids"], info["pv_values"], info["slot"], info["cost"], info["reward"]
            agents = info["agents"]
            na = len(agents)
            bp, pp = bids[:, p], pv[:, p]
            alpha_all = bids.sum(axis=0) / (pv.sum(axis=0) + 1e-12)       # 全広告主の実効入札係数（入札額の和/価値の和）
            sp = slot[p]
            st = ep_state.setdefault(ep, {"reward": np.zeros(na), "cost": np.zeros(na), "est": np.zeros(na),
                                          "real": np.zeros(na), "var": np.zeros(na), "exhaust": np.full(na, -1),
                                          "t0": time.time(), "cpu0": time.process_time(), "num_pv": 0,
                                          "trunc": [tuple(v) for v in info["trunc_values"]],
                                          "trunc_seeds": list(info["trunc_seeds"] or [None] * na),
                                          "bid_s": np.zeros(na), "n_drop": np.zeros(na), "n_iter": 0})
            st["reward"] += reward
            st["cost"] += cost
            st["num_pv"] += n
            rss = proc.memory_info().rss / 2 ** 20
            peak[0] = max(peak[0], rss)

            # ---- 全広告主の集計（列 = 広告主）。すべて、既にある配列からの計算で、乱数は引かない
            sig = info["pvalue_sigmas"]
            slotT = slot.T                                   # (機会, 広告主)
            exp = info["is_exposed"].T.astype(bool)
            won = slotT > 0
            preal = np.clip(info["values_real"], 0, 1)       # ②：雑音を加えて 0〜1 に切った確率
            vreal = info["values_real"]
            cost_pit = info["cost_pit"].T
            paid = cost_pit * exp
            conv = info["conversion"].T
            before = info["bids_before_adjust"]
            dropped = (before > 0) & (bids == 0)
            iters = info["sim_iters"]
            est_e = (pv * exp).sum(axis=0)
            real_e = (preal * exp).sum(axis=0)
            var_e = (preal * (1 - preal) * exp).sum(axis=0)
            st["est"] += est_e
            st["real"] += real_e
            st["var"] += var_e
            st["bid_s"] += (bids * exp).sum(axis=0)
            st["n_drop"] += dropped.sum(axis=0)
            st["n_iter"] += len(iters)
            rb_after = np.array([a.remaining_budget for a in agents], dtype=float)
            newly = (st["exhaust"] < 0) & (rb_after < min_remaining_budget)
            st["exhaust"][newly] = tick
            mp = info["market_prices"]
            sold = (slotT > 0).sum(axis=1)                   # 機会ごとの、売れた枠の数（0..3）
            env_s = {}
            for it in iters:
                for k, v in it["env_seconds"].items():
                    env_s[k] = env_s.get(k, 0.0) + v
            bq = _q(bp[:, None])[:, 0]
            pq = _q(pp[:, None])[:, 0]
            ex_p, won_p = exp[:, p], won[:, p]
            ia = info["internals"][p] or {}

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
                rss_mb=float(rss),
                # ---- ここから下は、記録の形式の版 2（T006）で足した列
                est_sum_exposed=float(est_e[p]), real_sum_exposed=float(real_e[p]), real_var_exposed=float(var_e[p]),
                est_sum_won=float(pp[won_p].sum()), real_sum_won=float(preal[won_p, p].sum()),
                est_sum_all=float(pp.sum()), real_sum_all=float(preal[:, p].sum()),
                real_var_all=float((preal[:, p] * (1 - preal[:, p])).sum()),
                conv_draw_sum_all=int(info["conversion_draw"][:, p].sum()),
                n_real_below0=int((vreal[:, p] < 0).sum()), n_real_above1=int((vreal[:, p] > 1).sum()),
                n_exposed_slot1=int((ex_p & (sp == 1)).sum()), n_exposed_slot2=int((ex_p & (sp == 2)).sum()),
                n_exposed_slot3=int((ex_p & (sp == 3)).sum()),
                cost_slot1=float(paid[sp == 1, p].sum()), cost_slot2=float(paid[sp == 2, p].sum()),
                cost_slot3=float(paid[sp == 3, p].sum()),
                reward_slot1=int(conv[sp == 1, p].sum()), reward_slot2=int(conv[sp == 2, p].sum()),
                reward_slot3=int(conv[sp == 3, p].sum()),
                bid_sum=float(bp.sum()), bid_sum_exposed=float(bp[ex_p].sum()), bid_sum_won=float(bp[won_p].sum()),
                bid_p10=float(bq[0]), bid_p50=float(bq[1]), bid_p90=float(bq[2]), bid_max=float(bp.max()) if n else 0.0,
                pvalue_p10=float(pq[0]), pvalue_p50=float(pq[1]), pvalue_p90=float(pq[2]),
                pvalue_max=float(pp.max()) if n else 0.0,
                pvalue_sigma_sum_exposed=float(sig[ex_p, p].sum()),
                alpha_internal=float("nan") if ia.get("alpha") is None else float(ia["alpha"]),
                remaining_budget_after=float(rb_after[p]),
                n_sim_iters=len(iters), n_bids_dropped=int(dropped[:, p].sum()),
                cost_first_iter=float(iters[0]["cost"][p]), reward_first_iter=float(iters[0]["reward"][p]),
                over_cost_ratio_first=float(iters[0]["over_cost_ratio"][p]),
                n_overcost_agents_first=int((iters[0]["over_cost_ratio"] > 0).sum()),
                mp1_mean=float(mp[:, 0].mean()), mp2_mean=float(mp[:, 1].mean()),
                lwc_p10=float(np.percentile(mp[:, 2], 10)), lwc_p50=float(np.percentile(mp[:, 2], 50)),
                lwc_p90=float(np.percentile(mp[:, 2], 90)),
                n_pv_sold3=int((sold == 3).sum()), n_pv_sold0=int((sold == 0).sum()),
                n_exposed_all=int(exp.sum()), reward_all=float(reward.sum()),
                bidding_seconds_all=float(info["bidding_seconds"].sum()),
                sim_seconds=float(sum(it["seconds"] for it in iters)), adjust_seconds=float(info["adjust_seconds"]),
                env_sort_seconds=env_s.get("sort"), env_slot_seconds=env_s.get("slot"),
                env_cost_seconds=env_s.get("cost"), env_exposure_seconds=env_s.get("exposure"),
                env_values_seconds=env_s.get("values"), env_conversion_seconds=env_s.get("conversion"),
                env_unsold_seconds=env_s.get("unsold"),
                tick_core_seconds=float(now - info["tick_begin"]), tick_cpu_seconds=float(cpu_now - last_cpu[0]),
                ended_at_epoch=time.time()))

            for k, it in enumerate(iters):
                oc = it["over_cost_ratio"]
                iter_rows.append(dict(episode=ep, tick=tick, iter=k, n_overcost_agents=int((oc > 0).sum()),
                                      over_cost_ratio_max=float(oc.max()), player_over_cost_ratio=float(oc[p]),
                                      player_cost=float(it["cost"][p]), player_reward=float(it["reward"][p]),
                                      total_cost_all=float(it["cost"].sum()), sim_seconds=float(it["seconds"]),
                                      overcost_agents=json.dumps(np.where(oc > 0)[0].tolist())))

            if level >= 1:
                bqa = _q(bids, (10, 50, 90))
                at = dict(
                    episode=np.full(na, ep), tick=np.full(na, tick), agent_index=np.arange(na),
                    is_player=np.arange(na) == p,
                    did_bid=np.array([x is not None for x in info["internals"]]) | (info["bidding_seconds"] > 0),
                    remaining_budget_before=info["remaining_budget_before"].astype(float), remaining_budget_after=rb_after,
                    alpha_eff=alpha_all,
                    alpha_internal=np.array([np.nan if (x is None or x.get("alpha") is None) else x["alpha"]
                                             for x in info["internals"]], dtype=float),
                    bid_sum=bids.sum(axis=0), bid_mean=bids.mean(axis=0), bid_p10=bqa[0], bid_p50=bqa[1], bid_p90=bqa[2],
                    bid_max=bids.max(axis=0), bid_nonzero_n=(bids > 0).sum(axis=0),
                    bid_sum_won=(bids * won).sum(axis=0), bid_sum_exposed=(bids * exp).sum(axis=0),
                    pvalue_sum=pv.sum(axis=0), pvalue_std=pv.std(axis=0), sigma_sum=sig.sum(axis=0),
                    n_won=won.sum(axis=0), n_slot1=(slotT == 1).sum(axis=0), n_slot2=(slotT == 2).sum(axis=0),
                    n_slot3=(slotT == 3).sum(axis=0), n_exposed=exp.sum(axis=0),
                    n_exposed_slot1=(exp & (slotT == 1)).sum(axis=0), n_exposed_slot2=(exp & (slotT == 2)).sum(axis=0),
                    n_exposed_slot3=(exp & (slotT == 3)).sum(axis=0),
                    cost=cost.astype(float), cost_slot1=(paid * (slotT == 1)).sum(axis=0),
                    cost_slot2=(paid * (slotT == 2)).sum(axis=0), cost_slot3=(paid * (slotT == 3)).sum(axis=0),
                    reward=reward.astype(float),
                    est_sum_exposed=est_e, real_sum_exposed=real_e, real_var_exposed=var_e,
                    est_sum_won=(pv * won).sum(axis=0), real_sum_won=(preal * won).sum(axis=0),
                    real_sum_all=preal.sum(axis=0), real_var_all=(preal * (1 - preal)).sum(axis=0),
                    conv_draw_sum_all=info["conversion_draw"].sum(axis=0),
                    n_real_below0=(vreal < 0).sum(axis=0), n_real_above1=(vreal > 1).sum(axis=0),
                    n_bids_dropped=dropped.sum(axis=0), cost_first_iter=iters[0]["cost"].astype(float),
                    over_cost_ratio_first=iters[0]["over_cost_ratio"].astype(float),
                    bidding_seconds=info["bidding_seconds"].astype(float))
                df = pd.DataFrame(at)
                ints = []
                for x in info["internals"]:
                    if x is None:
                        ints.append((None, None, None))
                        continue
                    rest = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in x.items()
                            if k not in ("state_raw", "state_norm")}
                    ints.append((None if "state_raw" not in x else [float(v) for v in x["state_raw"]],
                                 None if "state_norm" not in x else [float(v) for v in x["state_norm"]],
                                 json.dumps(rest)))
                df["state_raw"] = [i[0] for i in ints]
                df["state_norm"] = [i[1] for i in ints]
                df["internal_json"] = [i[2] for i in ints]
                agent_tick_frames.append(df)
                # プレイヤーが落札した機会と、予算超過で入札を取り消された機会
                pv_won_frames.append(pv_frame(info, np.where(won_p | dropped[:, p])[0]))
            if level >= 2:
                pv_all_frames.append(pv_frame(info, np.arange(n)))
            if level >= 3:
                b = pd.DataFrame(bids.astype(f32), columns=[f"bid_{i:02d}" for i in range(na)])
                b.insert(0, "pv_index", np.arange(n, dtype=np.int32))
                b.insert(0, "tick", np.int8(tick))
                b.insert(0, "episode", np.int16(ep))
                for i in range(3):
                    b[f"winner{i + 1}"] = info["winner"][:, i].astype(np.int8)
                b["exposed_slots"] = (exp * (1 << np.clip(slotT - 1, 0, 2))).sum(axis=1).astype(np.int8)   # 枠 1・2・3 の露出を 1・2・4 のビットで
                bids_all.setdefault(ep, []).append(b)
                if tick == info["num_tick_total"] - 1:
                    (tmp_dir / "raw").mkdir(exist_ok=True)
                    pd.concat(bids_all.pop(ep)).assign(run_id=run_id).to_parquet(
                        tmp_dir / "raw" / f"bids_all_ep{ep}.parquet", **PARQUET_KW)

            if tick == info["num_tick_total"] - 1:
                st["wall"] = time.time() - st["t0"]
                st["cpu"] = time.process_time() - st["cpu0"]
                for i, a in enumerate(agents):
                    agent_rows.append(dict(episode=ep, agent_index=i, agent_name=a.name, category=int(a.category),
                                           budget=float(a.budget), cpa_constraint=float(a.cpa),
                                           reward=float(st["reward"][i]), cost=float(st["cost"][i]),
                                           final_remaining_budget=float(a.remaining_budget),
                                           n_est=float(st["est"][i]), n_real=float(st["real"][i]),
                                           real_var=float(st["var"][i]),
                                           trunc_u1=float(st["trunc"][i][0]), trunc_u2=float(st["trunc"][i][1]),
                                           trunc_seed=st["trunc_seeds"][i],
                                           budget_exhausted_tick=int(st["exhaust"][i]),
                                           bid_sum_exposed=float(st["bid_s"][i]), n_bids_dropped=int(st["n_drop"][i]),
                                           strategy_class=type(getattr(a, "player_agent", a)).__name__
                                           if i != p else type(strategy._inner).__name__))
            t_end = time.perf_counter()
            tick_rows[-1]["hook_seconds"] = t_end - t_hook
            last_t[0] = t_end
            last_cpu[0] = time.process_time()

        # tick_hook に num_tick_total と sigma を渡すため、薄く包む
        num_tick = gin_value("run_test.num_tick")

        def hook2(info):
            info["num_tick_total"] = num_tick
            hook(info)

        t_sim = time.time()
        cpu_sim = time.process_time()
        res = run_test(player_index=p, player_agent=strategy, episode_ids=resolved["episodes"], tick_hook=hook2)
        sim_seconds = time.time() - t_sim
        sim_cpu_seconds = time.process_time() - cpu_sim

        # --- 集計 ---
        t_write = time.time()
        raw = pd.DataFrame(res["rawData"])
        eps, agent_frames = [], []
        agents_df = pd.DataFrame(agent_rows)
        ticks_df = pd.DataFrame(tick_rows)
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
            st = ep_state[ep]
            tk = ticks_df[ticks_df.episode == ep]
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
                others_score_mean=float(a[~a.is_player].score_component.mean()),
                # ---- 記録の形式の版 2（T006）で足した列
                n_est=float(st["est"][p]), n_real=float(st["real"][p]), real_var=float(st["var"][p]),
                n_won=int(tk.n_won.sum()), n_bids_dropped=int(st["n_drop"][p]), n_sim_iters=int(st["n_iter"]),
                trunc_u1=float(st["trunc"][p][0]), trunc_u2=float(st["trunc"][p][1]),
                budget_exhausted_tick=int(st["exhaust"][p]), num_pv=int(st["num_pv"]),
                episode_wall_seconds=float(st["wall"]), episode_cpu_seconds=float(st["cpu"]),
                bidding_seconds=float(tk.bidding_seconds.sum()), hook_seconds=float(tk.hook_seconds.sum())))
        pd.DataFrame(eps).assign(run_id=run_id).to_parquet(tmp_dir / "episodes.parquet", index=False)
        ticks_df.assign(run_id=run_id).to_parquet(tmp_dir / "ticks.parquet", index=False)
        pd.concat(agent_frames).assign(run_id=run_id).to_parquet(tmp_dir / "agents.parquet", index=False)
        pd.DataFrame(iter_rows).assign(run_id=run_id).to_parquet(tmp_dir / "sim_iters.parquet", **PARQUET_KW)
        if level >= 1:
            (tmp_dir / "raw").mkdir(exist_ok=True)
            pd.concat(agent_tick_frames).assign(run_id=run_id).to_parquet(tmp_dir / "raw" / "agent_ticks.parquet", **PARQUET_KW)
            pd.concat(pv_won_frames).assign(run_id=run_id).to_parquet(tmp_dir / "raw" / "pv_won.parquet", **PARQUET_KW)
        if level >= 2:
            pd.concat(pv_all_frames).assign(run_id=run_id).to_parquet(tmp_dir / "raw" / "pv_all.parquet", **PARQUET_KW)
        write_seconds = time.time() - t_write
        files = {str(f.relative_to(tmp_dir).as_posix()): f.stat().st_size for f in sorted(tmp_dir.rglob("*")) if f.is_file()}
        ru = resource.getrusage(resource.RUSAGE_SELF)
        try:
            load1 = os.getloadavg()
        except Exception:
            load1 = (None, None, None)
        meta.update(status="ok", score=float(res["score"]), reward_total=float(res["reward"]),
                    sim_seconds=sim_seconds, peak_rss_mb=peak[0],
                    bidding_seconds_total=float(sum(t["bidding_seconds"] for t in tick_rows)),
                    bidding_seconds_per_call_mean=float(np.mean([t["bidding_seconds"] for t in tick_rows])),
                    n_episodes=len(resolved["episodes"]), n_ticks=num_tick,
                    sim_cpu_seconds=sim_cpu_seconds, strategy_init_seconds=strategy_init_seconds,
                    write_seconds=write_seconds, hook_seconds_total=float(ticks_df.hook_seconds.sum()),
                    bidding_seconds_all_total=float(ticks_df.bidding_seconds_all.sum()),
                    env_sim_seconds_total=float(ticks_df.sim_seconds.sum()),
                    adjust_seconds_total=float(ticks_df.adjust_seconds.sum()),
                    n_sim_iters_total=int(ticks_df.n_sim_iters.sum()),
                    maxrss_mb=ru.ru_maxrss / 1024, cpu_user_seconds=ru.ru_utime, cpu_system_seconds=ru.ru_stime,
                    loadavg1_end=load1[0], mem_available_end_mb=psutil.virtual_memory().available / 2 ** 20,
                    gin_operative_config=gin.operative_config_str(),
                    files=json.dumps(files), bytes_total=int(sum(files.values())),
                    bytes_raw=int(sum(v for k, v in files.items() if k.startswith("raw/"))))
    except Exception as e:  # 止めずに記録して先へ
        meta.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    meta["wall_seconds"] = time.time() - t0
    meta["cpu_seconds"] = time.process_time() - cpu0
    meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
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


def build_tasks(spec_path, force, record=None):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if record:
        spec["record"] = record
    version = code_version()
    tasks, skipped = [], []
    for resolved in expand_spec(spec):
        resolved.setdefault("episodes", [0, 1])
        resolved.setdefault("seed", 1)
        resolved.setdefault("player_indices", [0, 1])
        resolved.setdefault("record", "standard")
        assert resolved["record"] in RECORD_LEVELS, f"record は {list(RECORD_LEVELS)} のどれか"
        assert resolved["player"]["strategy"] in STRATEGIES, f"strategy は {list(STRATEGIES)} のどれか"
        # 既定値と同じ指定は取り除く（同じ実効条件なら同じ run_id にして、既にある結果を使い回すため）
        defaults = strategy_defaults(resolved["player"]["strategy"])
        kw = resolved["player"].get("kwargs", {})
        resolved["player"]["kwargs"] = {k: v for k, v in kw.items() if not (k in defaults and defaults[k] == v)}
        resolved["spec_file"] = str(Path(spec_path).as_posix())
        for idx in resolved["player_indices"]:
            rid = make_run_id(resolved, idx, version["github_version"])
            mp = DB_RUNS / rid / "meta.json"
            old = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            if (old.get("status") == "ok" and not force
                    and old.get("record_level_num", 0) >= RECORD_LEVELS[resolved["record"]]):
                skipped.append(rid)  # status=ok で、記録の段が足りている run だけスキップ。失敗した run、段が低い run は流し直す
                continue
            tasks.append(dict(resolved=resolved, player_index=idx, run_id=rid, version=version))
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="既にある run を上書きする")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--runs-dir", help="出力先（既定は DB/runs）。容量の試し取りなど、DB に入れない実行に使う")
    ap.add_argument("--record", choices=list(RECORD_LEVELS), help="記録の段を、spec の record より優先して指定する")
    args = ap.parse_args()
    global DB_RUNS
    if args.runs_dir:
        DB_RUNS = Path(args.runs_dir).resolve()
    DB_RUNS.mkdir(parents=True, exist_ok=True)
    tasks, skipped = build_tasks(args.spec, args.force, args.record)
    for t in tasks:
        t.update(n_workers=args.workers, n_tasks=len(tasks), runs_dir=str(DB_RUNS))
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
