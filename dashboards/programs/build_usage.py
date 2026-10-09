#!/usr/bin/env python3
"""会話の記録（~/.claude/projects/）から、usage.html（Claude の使用量の画面）を作る。

トークンは 4 種類に分けて数える：入力（キャッシュ以外）、出力、キャッシュの読み出し、キャッシュへの書き込み。
桁が大きく違う（キャッシュの読み出しが大半）ので、1 つのグラフに積まず、種類ごとに別のグラフにする。
"""
import html
from collections import defaultdict
from datetime import datetime, timedelta

import transcripts as T
from common import CSS, JST, ROOT, nav, script_json, write_page

DAYS = 14                  # 日ごとの推移に出す日数
HOUR_DAYS = 7              # 時間帯別に使う日数
KINDS = [("output_tokens", "出力", "Claude が書いた量（考えた分を含む）"),
         ("input_tokens", "入力（キャッシュ以外）", "新しく読ませた量"),
         ("cache_read_input_tokens", "キャッシュの読み出し", "前のやり取りを読み直した量"),
         ("cache_creation_input_tokens", "キャッシュへの書き込み", "次に読み直すために保存した量")]


def esc(x):
    return html.escape(str(x if x is not None else ""))


def num(n):
    return f"{n / 1e9:.2f} G" if n >= 1e9 else f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n / 1e3:.1f} k" if n >= 1e3 else str(int(n))


def total(rows, key):
    return sum(r[key] for r in rows)


def table(groups, label):
    """groups: {名前: 応答のリスト}。出力の多い順に並べる。"""
    out = []
    for name, rows in sorted(groups.items(), key=lambda kv: -total(kv[1], "output_tokens")):
        out.append(f'<tr><td>{esc(name)}</td><td class="num">{len(rows):,}</td>'
                   + "".join(f'<td class="num">{num(total(rows, k))}</td>' for k, _, _ in KINDS) + "</tr>")
    head = f'<thead><tr><th>{label}</th><th class="num">応答数</th>' + "".join(f'<th class="num">{esc(n)}</th>' for _, n, _ in KINDS) + "</tr></thead>"
    return f'<table class="list">{head}<tbody>{"".join(out) or "<tr><td colspan=6>記録がありません</td></tr>"}</tbody></table>'


def main():
    now = datetime.now(JST)
    start_day = (now - timedelta(days=DAYS - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    rows, sessions = T.read_all(since=start_day)
    for r in rows:
        r["jst"] = r["when"].astimezone(JST)
    days = [(start_day + timedelta(days=i)).strftime("%m/%d") for i in range(DAYS)]
    by_day = {k: defaultdict(int) for k, _, _ in KINDS}
    by_hour = defaultdict(int)
    hour_since = now - timedelta(days=HOUR_DAYS)
    for r in rows:
        d = r["jst"].strftime("%m/%d")
        for k, _, _ in KINDS:
            by_day[k][d] += r[k]
        if r["jst"] >= hour_since:
            by_hour[r["jst"].hour] += r["output_tokens"]
    today = [r for r in rows if r["jst"].date() == now.date()]
    week = [r for r in rows if r["jst"] >= now - timedelta(days=7)]
    charts = {k: [[d, by_day[k][d]] for d in days] for k, _, _ in KINDS}
    charts["hour"] = [[f"{h:02d}", by_hour[h]] for h in range(24)]

    def group(key):
        g = defaultdict(list)
        for r in week:
            g[key(r)].append(r)
        return g

    def session_name(r):
        s = sessions.get(r["session_id"], {})
        p = " ".join((s.get("prompt") or "").split())
        return f'{r["jst"].strftime("%m/%d")}　{p[:44] or r["session_id"][:8]}'

    by_session = defaultdict(list)
    for r in week:
        by_session[r["session_id"]].append(r)
    sess_groups = {session_name(min(v, key=lambda r: r["when"])): v for v in by_session.values()}
    top_sessions = dict(sorted(sess_groups.items(), key=lambda kv: -total(kv[1], "output_tokens"))[:10])

    tiles = "".join(
        f'<div class="card tile"><div class="label">{esc(label)}</div><div class="num">{num(total(rs, "output_tokens"))}</div>'
        f'<div class="sub">出力トークン・応答 {len(rs):,} 回<br>キャッシュの読み出し {num(total(rs, "cache_read_input_tokens"))}</div></div>'
        for label, rs in (("今日", today), ("直近 7 日", week), (f"直近 {DAYS} 日", rows)))
    cards = "".join(f'<div class="card chart" data-chart="{k}" data-unit="トークン"><h3>{esc(n)}</h3><p>{esc(d)}・日ごと</p></div>' for k, n, d in KINDS)
    page = (TEMPLATE.replace("__CSS__", CSS).replace("__NAV__", nav("usage.html"))
            .replace("__UPDATED__", now.strftime("%Y-%m-%d %H:%M")).replace("__TILES__", tiles).replace("__CARDS__", cards)
            .replace("__DAYS__", str(DAYS)).replace("__HOUR_DAYS__", str(HOUR_DAYS))
            .replace("__BY_MODEL__", table(group(lambda r: r["model"]), "モデル"))
            .replace("__BY_PROJECT__", table(group(lambda r: r["project"].replace(str(ROOT.parent.parent), "~") or "（不明）"), "フォルダ"))
            .replace("__BY_ENTRY__", table(group(lambda r: {"claude-vscode": "VSCode", "sdk-cli": "CLI（-p）", "cli": "端末"}.get(r["entrypoint"], r["entrypoint"] or "（不明）")
                                                 + ("・サブエージェント" if r["subagent"] else "")), "起動のしかた"))
            .replace("__BY_SESSION__", table(top_sessions, "会話（始めた日・最初の指示）"))
            .replace("__DATA__", script_json(charts)))
    write_page(ROOT / "usage.html", page)
    print(f"{now.replace(microsecond=0).isoformat()} build usage.html responses={len(rows)} sessions={len(by_session)}")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude の使用量</title>
<style>
__CSS__</style>
</head>
<body>
<main>
  __NAV__
  <h1>Claude の使用量</h1>
  <p class="meta">最終更新 __UPDATED__（日本時間）
    <button type="button" id="refresh" hidden>更新</button>
    <span id="refresh-note"></span></p>

  <h2>まとめ</h2>
  <div class="tiles">__TILES__</div>

  <h2>日ごとの推移（直近 __DAYS__ 日）</h2>
  <div class="charts">__CARDS__</div>

  <h2>時間帯（直近 __HOUR_DAYS__ 日の合計）</h2>
  <div class="charts" style="grid-template-columns: 1fr">
    <div class="card chart" data-chart="hour" data-unit="トークン"><h3>出力トークン</h3><p>時刻（日本時間）ごとの合計</p></div>
  </div>

  <h2>内訳（直近 7 日）</h2>
  <div class="card scroll">__BY_MODEL__</div>
  <div class="card scroll" style="margin-top:12px">__BY_ENTRY__</div>
  <div class="card scroll" style="margin-top:12px">__BY_PROJECT__</div>

  <h2>出力の多い会話（直近 7 日、上位 10）</h2>
  <div class="card scroll">__BY_SESSION__</div>

  <ul class="notes">
    <li>もとは、Claude Code がこのサーバーに残す会話の記録（<span class="mono">~/.claude/projects/</span>）です。別の端末で動かした Claude のぶんは入りません。</li>
    <li>記録は、既定で 30 日たつと消えます。それより前の使用量は出せません。</li>
    <li>数えているのはトークンの数で、料金や、プランの上限に対する残りではありません。</li>
    <li>グラフは種類ごとに縦軸が違います（キャッシュの読み出しは、出力の 100 倍ほどになります）。</li>
  </ul>
</main>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
function el(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}
function short(n) {
  return n >= 1e9 ? (n / 1e9).toFixed(1) + " G" : n >= 1e6 ? (n / 1e6).toFixed(1) + " M" : n >= 1e3 ? (n / 1e3).toFixed(0) + " k" : String(Math.round(n));
}
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 5, 10]) if (v <= m * p) return m * p;
  return 10 * p;
}
function drawBars(card) {
  card.querySelectorAll("svg, .tip, .empty").forEach(n => n.remove());
  const pts = DATA[card.dataset.chart] || [];
  if (!pts.some(p => p[1] > 0)) {
    const note = document.createElement("div");
    note.className = "empty"; note.textContent = "この期間の記録がありません";
    card.appendChild(note); return;
  }
  const W = Math.max(card.clientWidth - 32, 240), H = 170, m = { l: 48, r: 8, t: 10, b: 22 };
  const yMax = niceMax(Math.max(...pts.map(p => p[1])));
  const band = (W - m.l - m.r) / pts.length, bw = Math.max(2, Math.min(band - 2, 28));
  const y = v => H - m.b - v / yMax * (H - m.t - m.b);
  const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });
  svg.setAttribute("aria-label", card.querySelector("h3").textContent);
  for (const v of [0, yMax / 2, yMax]) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = short(v);
  }
  const tip = document.createElement("div");
  tip.className = "tip";
  const every = Math.ceil(pts.length / Math.max(1, Math.floor((W - m.l - m.r) / 44)));
  pts.forEach((p, i) => {
    const cx = m.l + band * (i + 0.5);
    if (i % every === 0) el("text", { x: cx, y: H - 6, "text-anchor": "middle" }, svg).textContent = p[0];
    if (p[1] > 0) {
      const h = Math.max(1, H - m.b - y(p[1])), r = Math.min(4, bw / 2, h);
      // 上の角だけ丸める（下は基線に付ける）
      el("path", { class: "b", d: "M" + (cx - bw / 2) + " " + (H - m.b) + "v" + (-(h - r)) + "q0 " + (-r) + " " + r + " " + (-r) + "h" + (bw - 2 * r)
        + "q" + r + " 0 " + r + " " + r + "v" + (h - r) + "z", fill: "var(--series-1)" }, svg);
    }
    // 棒より広い当たり判定
    const hit = el("rect", { x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointerenter", () => {
      tip.textContent = "";
      const when = document.createElement("div"); when.className = "when"; when.textContent = p[0];
      const val = document.createElement("b"); val.textContent = Math.round(p[1]).toLocaleString() + " " + (card.dataset.unit || "");
      tip.appendChild(when); tip.appendChild(val);
      tip.style.display = "block";
      const px = svg.offsetLeft + cx / W * svg.getBoundingClientRect().width;
      tip.style.left = (px + tip.offsetWidth + 16 > card.clientWidth ? px - tip.offsetWidth - 10 : px + 10) + "px";
      tip.style.top = (svg.offsetTop + 8) + "px";
    });
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  });
  card.appendChild(svg); card.appendChild(tip);
}
function drawAll() { document.querySelectorAll(".chart").forEach(drawBars); }
window.addEventListener("resize", drawAll);
drawAll();
(function () {
  const button = document.getElementById("refresh"), note = document.getElementById("refresh-note");
  if (!location.protocol.startsWith("http")) { note.textContent = "・「更新」ボタンは、programs/serve.py を動かして http://localhost:8765/ で開くと使えます"; return; }
  button.hidden = false;
  button.addEventListener("click", async () => {
    button.disabled = true; button.textContent = "更新中…";
    try {
      const res = await fetch("refresh", { method: "POST", headers: { "X-Refresh": "1" } });
      if (!res.ok) throw new Error(await res.text());
      location.reload();
    } catch (e) { note.textContent = "更新に失敗しました（" + e.message + "）"; button.disabled = false; button.textContent = "更新"; }
  });
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
