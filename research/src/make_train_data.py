"""公開データ（dataset/traffic/period-N.csv）から、学習データ（rlData）を作る。

    .venv/bin/python research/src/make_train_data.py [--raw dataset/traffic] [--workers 2] [--periods 7 8 ...]

中身は本家の `TrainDataGenerator._generate_train_data`（1 行 = 広告主 1 社 × 1 ティック）をそのまま呼ぶ。
本家の `batch_generate_train_data` と違うのは、次の 3 点だけ。
- period ごとに別プロセスで並列に処理する（1 プロセス約 10GB。32GB の端末では 2 並列まで）。
- 結合する順を period 番号の昇順に固定する（本家は glob の順で、端末によって変わりうる）。
- period ごとの要約（行数、広告主ごとの予算・CPA 制約・平均の α など）を DB/train_data/ に残す。
  80GB の原本が無い端末でも、データの中身を確かめられるようにするため。
出力先は本家と同じ `<raw>/training_data_rlData_folder/`（period-N-rlData.csv と training_data_all-rlData.csv）。
"""
import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY_DIR = REPO / "DB" / "train_data"
ALL_NAME = "training_data_all-rlData.csv"

warnings.filterwarnings("ignore")


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def period_of(path):
    return int(re.search(r"period-(\d+)", Path(path).name).group(1))


def process_period(args):
    raw_path, out_dir = Path(args[0]), Path(args[1])
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    import pandas as pd
    from github.strategy_train_env.bidding_train_env.train_data_generator.train_data_generator import TrainDataGenerator
    n = period_of(raw_path)
    t0 = time.time()
    df = pd.read_csv(raw_path)
    exposed = df[df.isExposed == 1]
    g = df.groupby("advertiserNumber")
    adv = g.agg(category=("advertiserCategoryIndex", "first"), budget=("budget", "first"),
                cpa_constraint=("CPAConstraint", "first"), bid_sum=("bid", "sum"), pvalue_sum=("pValue", "sum"),
                n_won=("xi", "sum"), conversions=("conversionAction", "sum"))
    adv["alpha_mean"] = adv.bid_sum / adv.pvalue_sum
    adv["cost"] = exposed.groupby("advertiserNumber").cost.sum().reindex(adv.index).fillna(0.0)
    adv["real_cpa"] = adv.cost / adv.conversions.where(adv.conversions > 0)
    adv["alpha_tick0"] = (df[df.timeStepIndex == 0].groupby("advertiserNumber").bid.sum()
                          / df[df.timeStepIndex == 0].groupby("advertiserNumber").pValue.sum())
    summary = dict(
        period=n, raw_file=raw_path.name, raw_bytes=raw_path.stat().st_size, columns=list(df.columns),
        n_rows=int(len(df)), n_nan=int(df.isna().sum().sum()),
        # 欠損のある列と、その行の中身。公開データには、終盤のティックで入札額が NaN の行がある（その行は落札していない）
        nan_by_column={c: int(v) for c, v in df.isna().sum().items() if v > 0},
        nan_rows=dict(n_rows=int(df.isna().any(axis=1).sum()), n_won=float(df[df.isna().any(axis=1)].xi.sum()),
                      cost=float(df[df.isna().any(axis=1)].cost.sum()),
                      ticks=sorted(int(t) for t in df[df.isna().any(axis=1)].timeStepIndex.unique()),
                      advertisers=sorted(int(a) for a in df[df.isna().any(axis=1)].advertiserNumber.unique())),
        delivery_period_values=sorted(float(x) for x in df.deliveryPeriodIndex.unique()),
        n_advertisers=int(df.advertiserNumber.nunique()), n_ticks=int(df.timeStepIndex.nunique()),
        n_pv=int(len(df) // max(df.advertiserNumber.nunique(), 1)),
        pv_per_tick={int(k): int(v) for k, v in (df.groupby("timeStepIndex").size() // df.advertiserNumber.nunique()).items()},
        pvalue_mean=float(df.pValue.mean()), least_winning_cost_mean=float(df.leastWinningCost.mean()),
        bid_min=float(df.bid.min()), cost_min=float(df.cost.min()),
        won_slots_per_pv=float(df.xi.sum() / (len(df) / df.advertiserNumber.nunique())),
        advertisers=[dict(advertiser=int(i), **{k: (None if v != v else float(v)) for k, v in r.items()})
                     for i, r in adv.drop(columns=["bid_sum", "pvalue_sum"]).iterrows()])
    rl = TrainDataGenerator()._generate_train_data(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"period-{n}-rlData.csv"
    rl.to_csv(out_path, index=False)
    summary.update(rl_file=out_path.name, rl_rows=int(len(rl)), rl_sha1=sha1_of(out_path),
                   rl_done_rows=int((rl.done == 1).sum()), seconds=time.time() - t0)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / f"period-{n}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return n, len(rl), summary["seconds"]


def combine(out_dir, periods):
    import pandas as pd
    frames = [pd.read_csv(out_dir / f"period-{n}-rlData.csv") for n in sorted(periods)]
    all_path = out_dir / ALL_NAME
    pd.concat(frames, axis=0, ignore_index=True).to_csv(all_path, index=False)
    info = dict(file=ALL_NAME, periods=sorted(periods), rows=int(sum(len(f) for f in frames)), sha1=sha1_of(all_path),
                bytes=all_path.stat().st_size)
    (SUMMARY_DIR / "training_data_all.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO / "dataset" / "traffic"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--periods", type=int, nargs="*", help="処理する period（省略時は、ある period-N.csv を全部）")
    ap.add_argument("--force", action="store_true", help="既にある period も作り直す")
    a = ap.parse_args()
    raw = Path(a.raw)
    out_dir = raw / "training_data_rlData_folder"
    files = sorted(raw.glob("period-*.csv"), key=period_of)
    if a.periods:
        files = [f for f in files if period_of(f) in a.periods]
    todo = [f for f in files if a.force or not ((out_dir / f"period-{period_of(f)}-rlData.csv").exists()
                                                and (SUMMARY_DIR / f"period-{period_of(f)}.json").exists())]
    print(f"period {len(files)} 個 / 処理 {len(todo)} 個 / workers={a.workers}", flush=True)
    tasks = [(str(f), str(out_dir)) for f in todo]
    if a.workers <= 1 or len(tasks) <= 1:
        results = map(process_period, tasks)
    else:
        import multiprocessing as mp
        results = mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1).imap_unordered(process_period, tasks)
    for n, rows, sec in results:
        print(f"period-{n}: rlData {rows} 行 {sec:.0f}s", flush=True)
    # 結合は、--periods で絞ったときも、出来ている period を全部使う
    info = combine(out_dir, [period_of(f) for f in out_dir.glob("period-*-rlData.csv")])
    print("結合:", info, flush=True)


if __name__ == "__main__":
    main()
