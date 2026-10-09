"""spec.json から学習ベース手法（BC / IQL / CQL / BCQ / TD3_BC）を学習し、重みを DB/models/<model_id>/ に記録する。

    .venv/bin/python research/src/train_model.py <spec.json> [--workers 1] [--force] [--dry-run]

spec.json（変える値だけを書く）
    {
      "name": "train_default", "work": "W001", "rep": "REP002", "note": "自由記述",
      "algo": "BC",                      # BC / IQL / CQL / BCQ / TD3_BC
      "step_num": null,                  # 学習ステップ数。null（または書かない）なら本家の既定値
      "seed": 1,                         # random / numpy / torch の乱数シード
      "train_data": "DB/dataset/traffic/training_data_rlData_folder/training_data_all-rlData.csv",   # 省略可
      "checkpoints": [100, 1000, 10000],  # 任意。この学習ステップの時点の重みも ckpt/ に残す（学習の経過を見る用）
      "threads": 1,                      # 任意。torch のスレッド数（既定 1。BCQ は 8 で約 4 倍速い。数値が変わりうるので model_id に入る）
      "train_kwargs": {"max_action": 300},   # 任意。本家の学習関数にそのまま渡す追加の引数（例：BCQ の max_action）。書けば model_id に入る
      "sweep": {"algo": ["BC", "IQL"], "step_num": [null, 20000]}     # 任意。直積に展開する
    }

学習そのものは、本家の `github/strategy_train_env/run/run_*.py` の学習関数を、データの場所・保存先・ステップ数だけ渡して呼ぶ。
model_id は「手法・ステップ数・乱数シード・学習データの SHA-1・コードの版」から決まる。同じ条件なら同じ ID になり、既にあればスキップする。
DB/models/<model_id>/ に、重み（*.pth）、normalize_dict.pkl、meta.json（条件・コミット・所要時間・重みの SHA-1）、loss.csv（ステップごとの損失）を書く。
checkpoints を指定すると、DB/models/<model_id>/ckpt/<model_id>_s<ステップ>/ に、その時点の重み・normalize_dict.pkl・meta.json を書く
（途中で保存しても、最後の重みは変わらない。github/Test/2 の test_03 で確かめている）。チェックポイントも model_dir に指定して評価できる。
評価は run_experiment.py で、spec の player.kwargs.model_dir に "DB/models/<model_id>" を書いて行う。
"""
import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import platform
import random
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 学習も CPU で行う（ロックの torch 1.12.0 は新しい GPU で CUDA の計算が落ちる）

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import canonical, code_version, expand_spec  # noqa: E402

MODELS = REPO / "DB" / "models"
DEFAULT_DATA = "DB/dataset/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
# 手法 → (学習モジュール, 学習関数, 重みのファイル名)
ALGOS = {"BC": ("run_bc", "train_model", "bc_model.pth"), "IQL": ("run_iql", "train_iql_model", "iql_model.pth"),
         "CQL": ("run_cql", "train_cql_model", "cql_model.pth"), "BCQ": ("run_bcq", "train_bcq_model", "bcq_model.pth"),
         "TD3_BC": ("run_td3_bc", "train_td3_bc_model", "td3_bc_model.pth")}


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_sha1(path):
    """重みのテンソルそのものの SHA-1（名前の順に、形と中身をつなげたもの）。
    torch.jit の保存ファイルには、同じプロセスでそれまでに保存したモデルの数で変わる連番（___torch_mangle_N）が入るので、
    ファイルの SHA-1（model_sha1）は、重みが同じでも一致しないことがある。重みが同じかは、こちらで比べる。"""
    import torch
    sd = torch.jit.load(str(path), map_location="cpu").state_dict()
    h = hashlib.sha1()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def train_function(algo):
    import importlib
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    mod, fn, _ = ALGOS[algo]
    module = importlib.import_module(f"github.strategy_train_env.run.{mod}")
    return module, getattr(module, fn)


def default_step_num(algo):
    import inspect
    return inspect.signature(train_function(algo)[1]).parameters["step_num"].default


def make_model_id(resolved, version):
    key = {k: resolved[k] for k in ("algo", "step_num", "seed", "train_data_sha1")}
    if resolved.get("threads", 1) != 1:        # 既定（1 スレッド）の model_id は、これまでと変えない
        key["threads"] = resolved["threads"]
    if resolved.get("train_kwargs"):           # 追加の引数が無いときの model_id は、これまでと変えない
        key["train_kwargs"] = resolved["train_kwargs"]
    return "M" + hashlib.sha1((canonical(key) + "|" + version).encode("utf-8")).hexdigest()[:10]


class LossCapture(logging.Handler):
    """本家の学習ループが 1 ステップごとに出すログ（"Step: i Q_loss: x A_loss: y"）から、損失を拾う。"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.rows = []
        self.n_nonfinite = 0   # nan / inf になった損失の数（TD3_BC の A_loss は 2 ステップに 1 回しか出ない。出ない回は欠損で、ここには数えない）

    def emit(self, record):
        msg = record.getMessage()
        m = re.match(r"Step: (\d+)\s+(.*)", msg)
        if not m:
            return
        row = {"step": int(m.group(1))}
        for k, v in re.findall(r"([A-Za-z_ ]+?):\s*(-?[0-9.eE+-]+|nan|inf)", m.group(2)):
            row[k.strip().replace(" ", "_")] = float(v)
            self.n_nonfinite += int(float(v) != float(v) or abs(float(v)) == float("inf"))
        self.rows.append(row)


def install_checkpoints(module, steps, tmp_dir, model_id, meta, algo):
    """学習の 1 ステップ（model.step）を数え、指定のステップ数に達したら、その時点の重みを ckpt/ に保存する。元に戻す関数を返す。"""
    if not steps:
        return lambda: None
    cls = next(getattr(module, n) for n in ("BC", "IQL", "CQL", "BCQ", "TD3_BC") if hasattr(module, n))
    orig, count = cls.step, [0]

    def step(self, *a, **k):
        r = orig(self, *a, **k)
        count[0] += 1
        if count[0] in steps:
            cid = f"{model_id}_s{count[0]:06d}"
            d = tmp_dir / "ckpt" / cid
            self.save_jit(str(d))
            shutil.copy(tmp_dir / "normalize_dict.pkl", d / "normalize_dict.pkl")
            w = d / ALGOS[algo][2]
            (d / "meta.json").write_text(json.dumps(
                {"model_id": cid, "parent_model_id": model_id, "is_checkpoint": True, "algo": algo, "step_num": count[0],
                 "seed": meta["seed"], "train_kwargs": meta.get("train_kwargs") or {}, "spec_name": meta["spec_name"], "work": meta["work"], "rep": meta["rep"],
                 "train_data_sha1": meta["train_data_sha1"], "github_version": meta["github_version"],
                 "github_dirty": meta["github_dirty"], "head_commit": meta["head_commit"], "model_file": w.name,
                 "model_sha1": sha1_of(w), "weights_sha1": weights_sha1(w), "status": "ok"}, ensure_ascii=False, indent=2),
                encoding="utf-8")
        return r

    cls.step = step

    def restore():
        cls.step = orig
    return restore


def execute(task):
    resolved, model_id, version = task["resolved"], task["model_id"], task["version"]
    out_dir, tmp_dir = MODELS / model_id, MODELS / (model_id + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    algo = resolved["algo"]
    meta = {"model_id": model_id, "spec_name": resolved.get("name"), "work": resolved.get("work"), "rep": resolved.get("rep"),
            "note": resolved.get("note"), "sweep_point": resolved.get("sweep_point", {}), "algo": algo,
            "step_num": resolved["step_num"], "step_num_is_default": resolved["step_num_is_default"], "seed": resolved["seed"],
            "train_kwargs": resolved.get("train_kwargs") or {},
            "train_data": resolved["train_data"], "train_data_sha1": resolved["train_data_sha1"],
            "spec_file": resolved.get("spec_file"), "created_at": datetime.now().isoformat(timespec="seconds"),
            **{k: version[k] for k in ("head_commit", "github_tree", "github_dirty", "github_version", "repo_dirty_paths")},
            "host": platform.node(), "os": platform.platform(), "status": "running"}
    t0 = time.time()
    try:
        import numpy as np
        import pandas as pd
        import torch
        torch.set_num_threads(int(resolved.get("threads", 1)))
        module, fn = train_function(algo)
        cap = LossCapture()
        restore = install_checkpoints(module, sorted(set(resolved.get("checkpoints") or [])), tmp_dir, model_id, meta, algo)
        module.logger.addHandler(cap)
        module.logger.propagate = False        # 1 ステップごとのログを画面に出さない
        seed = resolved["seed"]
        random.seed(seed)                      # ReplayBuffer.sample は random.sample を使う（本家の main_*.py は固定していない）
        np.random.seed(seed)
        torch.manual_seed(seed)
        meta.update(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, pandas=pd.__version__,
                    device="cuda" if torch.cuda.is_available() else "cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            fn(train_data_path=str(REPO / resolved["train_data"]), save_path=str(tmp_dir), step_num=resolved["step_num"],
               **(resolved.get("train_kwargs") or {}))
        module.logger.removeHandler(cap)
        restore()
        loss = pd.DataFrame(cap.rows)
        loss.to_csv(tmp_dir / "loss.csv", index=False)
        weight = tmp_dir / ALGOS[algo][2]
        tail = loss.tail(max(1, len(loss) // 10)).drop(columns="step").mean().to_dict() if len(loss) else {}
        meta.update(status="ok", model_file=weight.name, model_sha1=sha1_of(weight), weights_sha1=weights_sha1(weight),
                    threads=int(resolved.get("threads", 1)), n_loss_rows=int(len(loss)),
                    checkpoints=sorted(int(d.name.rsplit("_s", 1)[1]) for d in (tmp_dir / "ckpt").glob("*_s*")) if (tmp_dir / "ckpt").is_dir() else [],
                    loss_last10pct_mean={k: float(v) for k, v in tail.items()},
                    loss_all_finite=cap.n_nonfinite == 0)
    except Exception as e:  # 止めずに記録して先へ
        meta.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    meta["train_seconds"] = time.time() - t0
    (tmp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return meta


def build_tasks(spec_path, force):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    version = code_version()
    sha_cache, tasks, skipped = {}, [], []
    for resolved in expand_spec(spec):
        assert resolved["algo"] in ALGOS, f"algo は {list(ALGOS)} のどれか"
        resolved.setdefault("seed", 1)
        resolved.setdefault("train_data", DEFAULT_DATA)
        d = default_step_num(resolved["algo"])
        resolved["step_num_is_default"] = resolved.get("step_num") in (None, d)
        resolved["step_num"] = d if resolved.get("step_num") is None else int(resolved["step_num"])
        data = REPO / resolved["train_data"]
        if resolved["train_data"] not in sha_cache:
            sha_cache[resolved["train_data"]] = sha1_of(data)
        resolved["train_data_sha1"] = sha_cache[resolved["train_data"]]
        resolved["spec_file"] = str(Path(spec_path).as_posix())
        resolved["checkpoints"] = sorted(c for c in set(resolved.get("checkpoints") or []) if c < resolved["step_num"])
        mid = make_model_id(resolved, version["github_version"])
        mp = MODELS / mid / "meta.json"
        if mp.exists() and not force and json.loads(mp.read_text(encoding="utf-8")).get("status") == "ok":
            skipped.append(mid)
            continue
        tasks.append(dict(resolved=resolved, model_id=mid, version=version))
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    tasks, skipped = build_tasks(a.spec, a.force)
    print(f"学習 {len(tasks)} 件 / 既存でスキップ {len(skipped)} 件 {skipped} / workers={a.workers}", flush=True)
    if a.dry_run:
        for t in tasks:
            print(t["model_id"], t["resolved"]["algo"], t["resolved"]["step_num"], t["resolved"].get("sweep_point"))
        return
    if a.workers <= 1 or len(tasks) <= 1:
        results = map(execute, tasks)
    else:
        import multiprocessing as mp
        results = mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1).imap_unordered(execute, tasks)
    for m in results:
        info = f"{m.get('train_seconds', 0):.0f}s {m.get('loss_last10pct_mean')}" if m["status"] == "ok" else m.get("error")
        print(f"{m['model_id']} {m['status']} {m['algo']} step={m['step_num']} {info}", flush=True)


if __name__ == "__main__":
    main()
