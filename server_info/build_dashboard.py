#!/usr/bin/env python3
"""履歴（history/*.csv）から dashboard.html を作る。

外部の CDN やフォントを読まない、1ファイルで完結した HTML を出す。
"""
import argparse
import csv
import html
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
JST = ZoneInfo("Asia/Tokyo")
CHART_DAYS = 7
TABLE_HOURS = 24
LEVELS = [("ok", "●", "余裕"), ("warn", "▲", "混雑"), ("crit", "■", "逼迫")]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def pct(part, whole):
    return None if part is None or not whole else round(100 * part / whole, 1)


def ms(ts):
    return int(datetime.fromisoformat(ts).timestamp() * 1000)


def level(value, bounds, below=False):
    """しきい値 [混雑, 逼迫] に対する段階（0=余裕, 1=混雑, 2=逼迫）。"""
    if value is None:
        return None
    if below:
        return sum(value < b for b in bounds)
    return sum(value >= b for b in bounds)


def badge(lv):
    if lv is None:
        return '<span class="badge na">取得できず</span>'
    cls, icon, label = LEVELS[lv]
    return f'<span class="badge {cls}"><span class="mark">{icon}</span>{label}</span>'


def fmt(value, unit="", digits=0):
    return "–" if value is None else f"{value:,.{digits}f}{unit}"


def gb(mb):
    return None if mb is None else mb / 1024


def host_metrics(r):
    cores, load15 = fnum(r["cpu_cores"]), fnum(r["load15"])
    total, avail = fnum(r["mem_total_mb"]), fnum(r["mem_available_mb"])
    return {
        "cores": cores, "load15": load15,
        "load_ratio": None if load15 is None or not cores else load15 / cores,
        "cpu_util": fnum(r["cpu_util_pct"]),
        "mem_total": total, "mem_avail": avail, "mem_avail_pct": pct(avail, total),
        "mem_used_pct": None if pct(avail, total) is None else round(100 - pct(avail, total), 1),
        "swap": fnum(r["swap_used_mb"]),
    }


def disk_used_pct(r):
    used, avail = fnum(r["used_gb"]), fnum(r["avail_gb"])
    return None if used is None or avail is None else pct(used, used + avail)


def current_rows(config, host, gpu, disk, latest):
    """現在値の表。サーバー1台を1行にする。最新の収集時刻の行だけを使う。"""
    th = config["thresholds"]
    latest_ts = latest.get("ts")
    out = []
    for server in config["servers"]:
        name = server["name"]
        info = latest.get("servers", {}).get(name, {})

        def at_latest(rows):
            return [r for r in rows if r["server"] == name and r["ts"] == latest_ts]

        h = at_latest(host)
        if h:
            m = host_metrics(h[0])
            cpu = (badge(level(m["load_ratio"], th["cpu_load_ratio"]))
                   + f'<div class="val">{fmt(pct(m["load15"], m["cores"]), "%")}</div>'
                   + f'<div class="sub">15分ロード {fmt(m["load15"], digits=2)} ÷ {fmt(m["cores"])} コア'
                   + f'<br>いまの使用率 {fmt(m["cpu_util"], "%")}</div>')
            mem = (badge(level(m["mem_avail_pct"], th["mem_available_pct_below"], below=True))
                   + f'<div class="val">空き {fmt(gb(m["mem_avail"]), " GB", 1)}</div>'
                   + f'<div class="sub">全 {fmt(gb(m["mem_total"]), " GB", 1)} のうち {fmt(m["mem_avail_pct"], "%")}'
                   + f'<br>スワップ使用 {fmt(gb(m["swap"]), " GB", 1)}</div>')
        else:
            cpu = mem = badge(None)

        gpu_cells = []
        for g in at_latest(gpu):
            used, total = fnum(g["mem_used_mb"]), fnum(g["mem_total_mb"])
            gpu_cells.append(
                badge(level(pct(used, total), th["vram_used_pct"]))
                + f'<div class="val">VRAM {fmt(gb(used), digits=1)} / {fmt(gb(total), " GB", 1)}</div>'
                + f'<div class="sub">GPU{html.escape(g["gpu_index"])} {html.escape(g["gpu_name"])}'
                + f'<br>使用率 {fmt(fnum(g["util_pct"]), "%")}・{fmt(fnum(g["temp_c"]), "℃")}</div>')
        procs = info.get("gpu_processes", [])
        if procs:
            items = "".join(
                f'<li>{html.escape(str(p["user"]))}・{fmt(gb(fnum(p["mem_used_mb"])), " GB", 1)}'
                f'・{html.escape(str(p["command"]))}（PID {html.escape(str(p["pid"]))}）</li>'
                for p in procs)
            gpu_cells.append(f'<div class="sub">使用中のプロセス</div><ul class="procs">{items}</ul>')
        elif gpu_cells:
            gpu_cells.append('<div class="sub">使用中のプロセスなし</div>')

        disk_cells = []
        for d in at_latest(disk):
            used = disk_used_pct(d)
            disk_cells.append(
                '<div class="disk">' + badge(level(used, th["disk_used_pct"]))
                + f'<div class="val">{html.escape(d["mount"])}　空き {fmt(fnum(d["avail_gb"]), " GB")}</div>'
                + f'<div class="sub">使用率 {fmt(used, "%")}（全 {fmt(fnum(d["total_gb"]), " GB")}）</div></div>')

        errors = "".join(f'<div class="err">取得失敗：{html.escape(e)}</div>'
                         for e in info.get("errors", []))
        out.append(
            f'<tr><th scope="row">{html.escape(name)}{errors}</th>'
            f'<td>{cpu}</td><td>{mem}</td>'
            f'<td>{"".join(gpu_cells) or badge(None)}</td>'
            f'<td>{"".join(disk_cells) or badge(None)}</td></tr>')
    return "\n".join(out)


def chart_data(config, host, gpu, disk, since):
    """グラフ用の系列。系列名は、区別が要るときだけサーバー名や GPU 番号を付ける。"""
    multi = len(config["servers"]) > 1
    series = {k: {} for k in ("load", "cpu", "mem", "gpu", "vram", "disk")}

    def add(chart, name, ts, value):
        series[chart].setdefault(name, []).append([ms(ts), value])

    for r in host:
        if r["ts"] < since:
            continue
        m = host_metrics(r)
        add("load", r["server"], r["ts"], None if m["load_ratio"] is None else round(100 * m["load_ratio"], 1))
        add("cpu", r["server"], r["ts"], m["cpu_util"])
        add("mem", r["server"], r["ts"], m["mem_used_pct"])

    gpus_per_server = {}
    for r in gpu:
        gpus_per_server.setdefault(r["server"], set()).add(r["gpu_index"])
    for r in gpu:
        if r["ts"] < since:
            continue
        name = r["server"] + (f' GPU{r["gpu_index"]}' if len(gpus_per_server[r["server"]]) > 1 else "")
        add("gpu", name, r["ts"], fnum(r["util_pct"]))
        add("vram", name, r["ts"], pct(fnum(r["mem_used_mb"]), fnum(r["mem_total_mb"])))

    for r in disk:
        if r["ts"] < since:
            continue
        name = (r["server"] + " " if multi else "") + r["mount"]
        add("disk", name, r["ts"], disk_used_pct(r))

    return {k: [{"name": n, "points": p} for n, p in v.items()] for k, v in series.items()}


def history_table(host, gpu, since):
    """直近の値の表（グラフの値を、ホバーなしでも読めるようにする）。"""
    gpu_at = {}
    for g in gpu:
        gpu_at.setdefault((g["ts"], g["server"]), []).append(g)
    rows = []
    for r in sorted((r for r in host if r["ts"] >= since), key=lambda r: r["ts"], reverse=True):
        m = host_metrics(r)
        gs = gpu_at.get((r["ts"], r["server"]), [])
        util = " / ".join(fmt(fnum(g["util_pct"]), "%") for g in gs) or "–"
        vram = " / ".join(fmt(pct(fnum(g["mem_used_mb"]), fnum(g["mem_total_mb"])), "%") for g in gs) or "–"
        when = datetime.fromisoformat(r["ts"]).astimezone(JST).strftime("%m/%d %H:%M")
        rows.append(
            f'<tr><td>{when}</td><td>{html.escape(r["server"])}</td>'
            f'<td>{fmt(pct(m["load15"], m["cores"]), "%")}</td><td>{fmt(m["cpu_util"], "%")}</td>'
            f'<td>{fmt(m["mem_used_pct"], "%")}</td><td>{util}</td><td>{vram}</td></tr>')
    return "\n".join(rows) or '<tr><td colspan="7">まだ履歴がありません</td></tr>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "servers.json")
    ap.add_argument("--data-dir", type=Path, default=HERE / "history")
    ap.add_argument("--out", type=Path, default=HERE / "dashboard.html")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    host = read_csv(args.data_dir / "host.csv")
    gpu = read_csv(args.data_dir / "gpu.csv")
    disk = read_csv(args.data_dir / "disk.csv")
    latest_path = args.data_dir / "latest.json"
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else {}

    now = datetime.now(JST)
    updated = datetime.fromisoformat(latest["ts"]) if latest.get("ts") else None
    data = {
        "updated": int(updated.timestamp() * 1000) if updated else None,
        "interval_min": config["interval_minutes"],
        "stale_min": config["stale_minutes"],
        "charts": chart_data(config, host, gpu, disk, (now - timedelta(days=CHART_DAYS)).isoformat()),
    }
    th = config["thresholds"]
    page = (TEMPLATE
            .replace("__UPDATED__", updated.strftime("%Y-%m-%d %H:%M") if updated else "まだ収集されていません")
            .replace("__INTERVAL__", str(config["interval_minutes"]))
            .replace("__STALE__", str(config["stale_minutes"]))
            .replace("__CURRENT__", current_rows(config, host, gpu, disk, latest))
            .replace("__HISTORY__", history_table(host, gpu, (now - timedelta(hours=TABLE_HOURS)).isoformat()))
            .replace("__TH_CPU__", f'{th["cpu_load_ratio"][0]:.0%} 以上で混雑、{th["cpu_load_ratio"][1]:.0%} 以上で逼迫')
            .replace("__TH_MEM__", f'空きが {th["mem_available_pct_below"][0]}% 未満で混雑、{th["mem_available_pct_below"][1]}% 未満で逼迫')
            .replace("__TH_VRAM__", f'{th["vram_used_pct"][0]}% 以上で混雑、{th["vram_used_pct"][1]}% 以上で逼迫')
            .replace("__TH_DISK__", f'{th["disk_used_pct"][0]}% 以上で混雑、{th["disk_used_pct"][1]}% 以上で逼迫')
            # </script> で閉じられないよう、JSON 中の < をエスケープする
            .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")))

    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"{now.replace(microsecond=0).isoformat()} build {args.out.name} host_rows={len(host)}")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>サーバー負荷</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--page); color: var(--ink);
  font: 14px/1.6 system-ui, -apple-system, "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif; }
main { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0; }
h2 { font-size: 16px; margin: 32px 0 12px; }
.meta { color: var(--ink2); margin: 4px 0 0; }
.stale { display: none; margin: 16px 0 0; padding: 12px 16px; border-radius: 8px;
  border: 2px solid var(--critical); background: var(--surface); font-weight: 600; }
.stale.on { display: block; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; vertical-align: top; padding: 12px 16px; border-bottom: 1px solid var(--grid); }
thead th { font-size: 12px; font-weight: 600; color: var(--ink2); white-space: nowrap; }
tbody tr:last-child > * { border-bottom: 0; }
.current td { min-width: 190px; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
  padding: 1px 8px; border-radius: 999px; border: 1px solid var(--border); }
.badge .mark { font-size: 11px; }
.badge.ok .mark { color: var(--good); }
.badge.warn .mark { color: var(--warning); }
.badge.crit .mark { color: var(--critical); }
.badge.na { color: var(--muted); }
.val { font-size: 16px; font-weight: 600; margin-top: 4px; }
.sub { color: var(--ink2); font-size: 12px; }
.disk + .disk { margin-top: 12px; }
.procs { margin: 2px 0 0; padding-left: 18px; font-size: 12px; color: var(--ink2); }
.err { color: var(--ink2); font-size: 12px; font-weight: 400; }
.filters { display: flex; align-items: center; gap: 8px; margin: 0 0 12px; }
.filters button { font: inherit; color: var(--ink2); background: transparent; cursor: pointer;
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 12px; }
.filters button[aria-pressed="true"] { color: var(--ink); font-weight: 600; background: var(--surface);
  border-color: var(--axis); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
.chart { padding: 12px 16px 8px; position: relative; min-width: 0; }
.chart h3 { font-size: 14px; margin: 0; }
.chart p { margin: 0 0 4px; color: var(--ink2); font-size: 12px; }
.chart svg { display: block; width: 100%; height: 170px; overflow: visible; }
.chart text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--ink2); }
.key { display: inline-block; width: 14px; height: 2px; border-radius: 1px; vertical-align: middle; margin-right: 6px; }
.tip { position: absolute; pointer-events: none; display: none; z-index: 1; background: var(--surface);
  border: 1px solid var(--axis); border-radius: 6px; padding: 6px 10px; font-size: 12px; white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.tip .when { color: var(--ink2); }
.tip b { font-variant-numeric: tabular-nums; margin-right: 6px; }
.empty { color: var(--muted); font-size: 12px; }
details { margin-top: 24px; }
summary { cursor: pointer; color: var(--ink2); }
.history td, .history th { padding: 6px 16px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.notes { color: var(--ink2); font-size: 12px; margin-top: 24px; padding-left: 18px; }
</style>
</head>
<body>
<main>
  <h1>サーバー負荷</h1>
  <p class="meta">最終更新 __UPDATED__（日本時間）・__INTERVAL__ 分ごとに更新・開き直すと最新になります</p>
  <div class="stale" id="stale" role="alert"></div>

  <h2>いまの状態</h2>
  <div class="card scroll">
    <table class="current">
      <thead><tr><th>サーバー</th><th>CPU（15分ロード ÷ コア数）</th><th>メモリ</th><th>GPU</th><th>ディスク</th></tr></thead>
      <tbody>
__CURRENT__
      </tbody>
    </table>
  </div>

  <h2>推移</h2>
  <div class="filters" role="group" aria-label="表示期間">
    <button type="button" data-hours="24" aria-pressed="true">直近24時間</button>
    <button type="button" data-hours="168" aria-pressed="false">直近7日</button>
  </div>
  <div class="charts">
    <div class="card chart" data-chart="load"><h3>CPU の混み具合</h3><p>15分ロードアベレージ ÷ コア数</p></div>
    <div class="card chart" data-chart="cpu"><h3>CPU 使用率</h3><p>収集時点の数秒間の平均</p></div>
    <div class="card chart" data-chart="mem"><h3>メモリ使用率</h3><p>キャッシュを除く（100% − 利用可能量）</p></div>
    <div class="card chart" data-chart="gpu"><h3>GPU 使用率</h3><p>収集時点の値</p></div>
    <div class="card chart" data-chart="vram"><h3>VRAM 使用率</h3><p>使用量 ÷ 総量</p></div>
    <div class="card chart" data-chart="disk" data-fixed-hours="168"><h3>ディスク使用率</h3><p>常に直近7日を表示</p></div>
  </div>

  <details>
    <summary>直近24時間の値を表で見る</summary>
    <div class="card scroll" style="margin-top:12px">
      <table class="history">
        <thead><tr><th>時刻</th><th>サーバー</th><th>CPU の混み具合</th><th>CPU 使用率</th><th>メモリ使用率</th><th>GPU 使用率</th><th>VRAM 使用率</th></tr></thead>
        <tbody>
__HISTORY__
        </tbody>
      </table>
    </div>
  </details>

  <ul class="notes">
    <li>CPU：15分ロード ÷ コア数が __TH_CPU__</li>
    <li>メモリ：__TH_MEM__</li>
    <li>VRAM：使用率が __TH_VRAM__</li>
    <li>ディスク：使用率が __TH_DISK__</li>
  </ul>
</main>

<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const JST_MS = 9 * 3600 * 1000;
const HOUR = 3600 * 1000;

function el(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}
function pad(n) { return String(n).padStart(2, "0"); }
function jst(t) { return new Date(t + JST_MS); }
function fmtWhen(t) {
  const d = jst(t);
  return (d.getUTCMonth() + 1) + "/" + d.getUTCDate() + " " + pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes());
}

// 更新が止まっていないかを、開いた時点の時刻で判定する
(function () {
  const box = document.getElementById("stale");
  if (DATA.updated === null) {
    box.textContent = "まだ一度も収集されていません。";
    box.classList.add("on");
    return;
  }
  const minutes = Math.floor((Date.now() - DATA.updated) / 60000);
  if (minutes >= DATA.stale_min) {
    const text = minutes >= 120 ? Math.floor(minutes / 60) + " 時間" : minutes + " 分";
    box.textContent = "■ 更新が止まっています。最後の更新から " + text + " たっています。下の値は古い可能性があります。";
    box.classList.add("on");
  }
})();

function drawChart(card, hours) {
  card.querySelectorAll("svg, .legend, .tip, .empty").forEach(n => n.remove());
  const end = DATA.updated === null ? Date.now() : Math.max(DATA.updated, Date.now() - DATA.stale_min * 60000);
  const start = end - hours * HOUR;
  const series = (DATA.charts[card.dataset.chart] || []).map((s, i) => ({
    name: s.name,
    color: "var(--series-" + (i % 8 + 1) + ")",
    points: s.points.filter(p => p[0] >= start && p[0] <= end),
  }));
  const values = series.flatMap(s => s.points.map(p => p[1])).filter(v => v !== null);
  if (!values.length) {
    const note = document.createElement("div");
    note.className = "empty";
    note.textContent = "この期間のデータがありません";
    card.appendChild(note);
    return;
  }

  if (series.length > 1) {
    const legend = document.createElement("div");
    legend.className = "legend";
    for (const s of series) {
      const item = document.createElement("span");
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = s.color;
      item.appendChild(key);
      item.appendChild(document.createTextNode(s.name));
      legend.appendChild(item);
    }
    card.appendChild(legend);
  }

  const W = Math.max(card.clientWidth - 32, 240), H = 170;
  const m = { l: 40, r: 12, t: 10, b: 22 };
  const yMax = Math.max(100, Math.ceil(Math.max(...values) / 50) * 50);
  const x = t => m.l + (t - start) / (end - start) * (W - m.l - m.r);
  const y = v => H - m.b - v / yMax * (H - m.t - m.b);
  const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });
  svg.setAttribute("aria-label", card.querySelector("h3").textContent + "の推移");

  for (const v of [0, yMax / 2, yMax]) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = v + "%";
  }
  // 横軸の目盛り：24時間なら6時間ごと、7日なら1日ごと（日本時間）
  const step = hours <= 48 ? 6 * HOUR : 24 * HOUR;
  for (let t = Math.ceil((start + JST_MS) / step) * step - JST_MS; t <= end; t += step) {
    const d = jst(t);
    const label = hours <= 48 ? pad(d.getUTCHours()) + ":00" : (d.getUTCMonth() + 1) + "/" + d.getUTCDate();
    el("line", { x1: x(t), x2: x(t), y1: H - m.b, y2: H - m.b + 4, stroke: "var(--axis)", "stroke-width": 1 }, svg);
    el("text", { x: x(t), y: H - 6, "text-anchor": "middle" }, svg).textContent = label;
  }

  // 収集が抜けた区間は線をつながない
  const gap = DATA.interval_min * 60000 * 2.5;
  for (const s of series) {
    let seg = [];
    const segs = [];
    let prev = null;
    for (const p of s.points) {
      if (p[1] === null || (prev !== null && p[0] - prev > gap)) { if (seg.length) segs.push(seg); seg = []; }
      if (p[1] !== null) { seg.push(p); prev = p[0]; }
    }
    if (seg.length) segs.push(seg);
    for (const g of segs) {
      if (g.length === 1) {
        el("circle", { cx: x(g[0][0]), cy: y(g[0][1]), r: 2, fill: s.color }, svg);
      } else {
        el("path", { d: g.map((p, i) => (i ? "L" : "M") + x(p[0]).toFixed(1) + " " + y(p[1]).toFixed(1)).join(""),
          fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      }
    }
    const last = segs.length ? segs[segs.length - 1].slice(-1)[0] : null;
    if (last) el("circle", { cx: x(last[0]), cy: y(last[1]), r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
  }

  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
  const tip = document.createElement("div");
  tip.className = "tip";
  card.appendChild(svg);
  card.appendChild(tip);

  const times = [...new Set(series.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
  svg.addEventListener("pointermove", e => {
    const box = svg.getBoundingClientRect();
    const t = start + ((e.clientX - box.left) / box.width * W - m.l) / (W - m.l - m.r) * (end - start);
    const near = times.reduce((a, b) => Math.abs(b - t) < Math.abs(a - t) ? b : a);
    cross.setAttribute("x1", x(near)); cross.setAttribute("x2", x(near));
    cross.setAttribute("visibility", "visible");
    tip.textContent = "";
    const when = document.createElement("div");
    when.className = "when";
    when.textContent = fmtWhen(near);
    tip.appendChild(when);
    for (const s of series) {
      const p = s.points.find(q => q[0] === near);
      const row = document.createElement("div");
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = s.color;
      const value = document.createElement("b");
      value.textContent = p && p[1] !== null ? p[1].toFixed(1) + "%" : "–";
      row.appendChild(key); row.appendChild(value);
      if (series.length > 1) row.appendChild(document.createTextNode(s.name));
      tip.appendChild(row);
    }
    tip.style.display = "block";
    const px = svg.offsetLeft + x(near) / W * box.width;
    const flip = px + tip.offsetWidth + 16 > card.clientWidth;
    tip.style.left = (flip ? px - tip.offsetWidth - 10 : px + 10) + "px";
    tip.style.top = (svg.offsetTop + 8) + "px";
  });
  svg.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; });
}

let hours = 24;
function drawAll() {
  document.querySelectorAll(".chart").forEach(card => drawChart(card, Number(card.dataset.fixedHours) || hours));
}
document.querySelectorAll(".filters button").forEach(button => {
  button.addEventListener("click", () => {
    hours = Number(button.dataset.hours);
    document.querySelectorAll(".filters button").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
    drawAll();
  });
});
window.addEventListener("resize", drawAll);
drawAll();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
