"""MOPO・COMBO（同梱の重み）を、プレイヤーとして評価する（参考。REP.md 3.3 節）。

    .venv/bin/python <このREP>/programs/run_bundled_mbrl.py <spec.json> [--workers 3] [--dry-run]

research/src/run_experiment.py は、この 2 手法を戦略の表（STRATEGIES）に持たない。共有のコードを書き換えずに済むよう、
このスクリプトの中でだけ表に足して、run_experiment.py の実行部（build_tasks・execute_run）をそのまま呼ぶ。
戦略のクラスは、本家のシミュレータが背景の広告主に使っているもの（github/simul_bidding_env/strategy/mbrl_*.py）で、変えていない。
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research" / "src"))
import run_experiment as rx  # noqa: E402

EXTRA = {"MOPO": ("mbrl_mopo_bidding_strategy", "MbrlMopoBiddingStrategy"),
         "COMBO": ("mbrl_combomicro_bidding_strategy", "MbrlComboMicroBiddingStrategy")}


def _patch():
    rx.STRATEGIES.update(EXTRA)


def _run(task):
    _patch()
    return rx.execute_run(task)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _patch()
    tasks, skipped = rx.build_tasks(args.spec, False)
    print(f"実行 {len(tasks)} 件 / 既存でスキップ {len(skipped)} 件 / workers={args.workers}", flush=True)
    if args.dry_run:
        for t in tasks:
            print(t["run_id"], t["resolved"]["player"])
        sys.exit(0)
    import multiprocessing as mp
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for m in pool.imap_unordered(_run, tasks):
            done += 1
            info = f"score={m.get('score', float('nan')):.5f} sim={m.get('sim_seconds', 0):.0f}s" if m["status"] == "ok" else m.get("error")
            print(f"[{done}/{len(tasks)}] {m['run_id']} {m['status']} {m['strategy']} p{m['player_index']} {info}", flush=True)
    print(f"完了 {time.time() - t0:.0f}s", flush=True)
