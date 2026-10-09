#!/usr/bin/env python3
"""サーバーの負荷を1回分だけ集めて、履歴（CSV）に追記する。

標準ライブラリと nvidia-smi だけで動く。一部の取得に失敗しても、
取れた項目は記録し、失敗した項目は空欄（欠損）として残す。
"""
import argparse
import csv
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # dashboards/（history/ を置く）
JST = ZoneInfo("Asia/Tokyo")
CPU_SAMPLE_SECONDS = 3

HOST_COLS = ["ts", "server", "cpu_cores", "cpu_util_pct", "load1", "load5", "load15",
             "mem_total_mb", "mem_available_mb", "swap_used_mb"]
GPU_COLS = ["ts", "server", "gpu_index", "gpu_name", "util_pct",
            "mem_used_mb", "mem_total_mb", "temp_c"]
DISK_COLS = ["ts", "server", "mount", "total_gb", "used_gb", "avail_gb"]


def cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:9]]
    return v[3] + v[4], sum(v)  # idle + iowait, 合計


def collect_host():
    idle0, total0 = cpu_times()
    time.sleep(CPU_SAMPLE_SECONDS)
    idle1, total1 = cpu_times()
    util = 100 * (1 - (idle1 - idle0) / max(total1 - total0, 1))

    with open("/proc/loadavg") as f:
        load1, load5, load15 = f.read().split()[:3]

    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val = line.split(":")
            mem[key] = int(val.split()[0])  # kB
    return {
        "cpu_cores": os.cpu_count(),
        "cpu_util_pct": round(util, 1),
        "load1": load1, "load5": load5, "load15": load15,
        "mem_total_mb": mem["MemTotal"] // 1024,
        "mem_available_mb": mem["MemAvailable"] // 1024,
        "swap_used_mb": (mem["SwapTotal"] - mem["SwapFree"]) // 1024,
    }


def nvidia_smi(query_flag, fields):
    out = subprocess.run(
        ["nvidia-smi", f"{query_flag}={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=True).stdout
    return [dict(zip(fields, [c.strip() for c in line.split(",")]))
            for line in out.splitlines() if line.strip()]


def num(text):
    """nvidia-smi は取れない値を [N/A] 等で返す。数値でなければ欠損にする。"""
    try:
        return float(text)
    except ValueError:
        return ""


def collect_gpu():
    gpus = nvidia_smi("--query-gpu", ["index", "uuid", "name", "utilization.gpu",
                                      "memory.used", "memory.total", "temperature.gpu"])
    rows = [{"gpu_index": g["index"], "gpu_name": g["name"],
             "util_pct": num(g["utilization.gpu"]), "mem_used_mb": num(g["memory.used"]),
             "mem_total_mb": num(g["memory.total"]), "temp_c": num(g["temperature.gpu"])}
            for g in gpus]
    index_of = {g["uuid"]: g["index"] for g in gpus}

    procs = []
    for p in nvidia_smi("--query-compute-apps", ["pid", "gpu_uuid", "used_memory"]):
        try:
            user = pwd.getpwuid(os.stat(f"/proc/{p['pid']}").st_uid).pw_name
            with open(f"/proc/{p['pid']}/comm") as f:
                command = f.read().strip()
        except (OSError, KeyError):
            user, command = "?", "?"
        procs.append({"gpu_index": index_of.get(p["gpu_uuid"], "?"), "pid": p["pid"],
                      "user": user, "mem_used_mb": num(p["used_memory"]), "command": command})
    return rows, procs


def collect_disks(mounts):
    rows = []
    for mount in mounts:
        u = shutil.disk_usage(mount)
        # df と同じく、使用率は「使用 ÷（使用＋一般ユーザーが使える空き）」で見る
        rows.append({"mount": mount, "total_gb": round(u.total / 2**30, 1),
                     "used_gb": round(u.used / 2**30, 1), "avail_gb": round(u.free / 2**30, 1)})
    return rows


def collect_local(server):
    """1台分を集める。戻り値は (host行 or None, gpu行, disk行, gpuプロセス, エラー)。"""
    host, gpus, disks, procs, errors = None, [], [], [], []
    try:
        host = collect_host()
    except Exception as e:
        errors.append(f"CPU・メモリ: {e}")
    try:
        gpus, procs = collect_gpu()
    except Exception as e:
        errors.append(f"GPU: {e}")
    for mount in server.get("disks", []):
        try:
            disks += collect_disks([mount])
        except Exception as e:
            errors.append(f"ディスク {mount}: {e}")
    return host, gpus, disks, procs, errors


def append_and_prune(path, cols, new_rows, cutoff):
    """追記し、保持期間より古い行を落とす。一時ファイルに書いてから置き換える。"""
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("ts", "") >= cutoff]
    rows += new_rows
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "servers.json")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "history")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    args.data_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(JST).replace(microsecond=0)
    ts = now.isoformat()
    cutoff = (now - timedelta(days=config["retention_days"])).isoformat()

    host_rows, gpu_rows, disk_rows, latest = [], [], [], {}
    for server in config["servers"]:
        name = server["name"]
        if server.get("method", "local") != "local":
            latest[name] = {"gpu_processes": [], "errors": [f"未対応の取得方法: {server['method']}"]}
            continue
        host, gpus, disks, procs, errors = collect_local(server)
        key = {"ts": ts, "server": name}
        if host:
            host_rows.append({**key, **host})
        gpu_rows += [{**key, **g} for g in gpus]
        disk_rows += [{**key, **d} for d in disks]
        latest[name] = {"gpu_processes": procs, "errors": errors}

    append_and_prune(args.data_dir / "host.csv", HOST_COLS, host_rows, cutoff)
    append_and_prune(args.data_dir / "gpu.csv", GPU_COLS, gpu_rows, cutoff)
    append_and_prune(args.data_dir / "disk.csv", DISK_COLS, disk_rows, cutoff)
    tmp = args.data_dir / "latest.tmp"
    tmp.write_text(json.dumps({"ts": ts, "servers": latest}, ensure_ascii=False, indent=1))
    os.replace(tmp, args.data_dir / "latest.json")

    all_errors = [f"{n}: {e}" for n, v in latest.items() for e in v["errors"]]
    print(f"{ts} collect host={len(host_rows)} gpu={len(gpu_rows)} disk={len(disk_rows)}"
          + (f" errors={all_errors}" if all_errors else ""))
    return 1 if not host_rows and not gpu_rows and not disk_rows else 0


if __name__ == "__main__":
    sys.exit(main())
