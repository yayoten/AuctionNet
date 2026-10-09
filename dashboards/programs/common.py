"""3 つの画面（サーバー負荷・エージェント・使用量）で共通の、置き場所・見た目・書き出し。"""
import html
import os
from pathlib import Path
from zoneinfo import ZoneInfo

PROGRAMS = Path(__file__).resolve().parent
ROOT = PROGRAMS.parent                 # dashboards/（画面の HTML、history/、logs/、agents/ を置く）
REPO = ROOT.parent
JST = ZoneInfo("Asia/Tokyo")
PAGES = [("server.html", "サーバー負荷"), ("agents.html", "AI エージェント"), ("usage.html", "Claude の使用量")]


def nav(current):
    """画面の上に置く切り替え。いま開いている画面には aria-current を付ける。"""
    items = "".join(
        f'<a href="{href}"' + (' aria-current="page"' if href == current else "") + f">{html.escape(label)}</a>"
        for href, label in PAGES)
    return f'<nav class="tabs" aria-label="画面の切り替え">{items}</nav>'


def write_page(path, text):
    """生成途中の画面を見せないよう、一時ファイルに書いてから置き換える。"""
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def script_json(data):
    """<script> の中に埋める JSON。</script> で閉じられないよう、< をエスケープする。"""
    import json
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


CSS = r""":root {
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
.meta button { font: inherit; font-weight: 600; color: var(--ink); background: var(--surface); cursor: pointer;
  border: 1px solid var(--axis); border-radius: 6px; padding: 4px 16px; margin: 0 8px; }
.meta button:disabled { color: var(--muted); cursor: default; }
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
.tabs { display: flex; gap: 4px; margin: 0 0 20px; border-bottom: 1px solid var(--grid); }
.tabs a { padding: 8px 14px; color: var(--ink2); text-decoration: none; border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tabs a[aria-current="page"] { color: var(--ink); font-weight: 600; border-bottom-color: var(--series-1); }
.tabs a:hover { color: var(--ink); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { padding: 12px 16px; }
.tile .label { color: var(--ink2); font-size: 12px; }
.tile .num { font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .sub { margin-top: 2px; }
.list td, .list th { padding: 8px 12px; font-size: 13px; }
.list td.num, .list th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.list .nowrap { white-space: nowrap; }
.pill { display: inline-block; font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 999px;
  border: 1px solid var(--border); white-space: nowrap; }
.pill .mark { font-size: 11px; margin-right: 4px; }
.pill.run .mark { color: var(--good); }
.pill.wait .mark { color: var(--warning); }
.pill.stop .mark { color: var(--critical); }
.pill.done .mark { color: var(--muted); }
.bar { height: 6px; border-radius: 3px; background: var(--grid); overflow: hidden; min-width: 80px; }
.bar > i { display: block; height: 100%; background: var(--series-1); border-radius: 3px; }
.warnbox { margin: 16px 0 0; padding: 12px 16px; border-radius: 8px; border: 2px solid var(--warning); background: var(--surface); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.files { margin: 2px 0 0; padding-left: 16px; font-size: 12px; color: var(--ink2); }
.chart rect.b { fill: var(--series-1); }
.chart rect.b:hover { opacity: 0.8; }
"""
