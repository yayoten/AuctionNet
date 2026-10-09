#!/usr/bin/env python3
"""台帳（agents/）と会話の記録から、agents.html（AI エージェントの画面）を作る。

上から、(1) 数の要約、(2) タスク（予約）の一覧、(3) 動いている Claude の一覧。
タスクは人か Claude が agentctl.py で作るもの、Claude の一覧はフックと会話の記録から自動で出るもの。
"""
import html
import os
import subprocess
from datetime import datetime, timedelta

import agents_lib as A
import transcripts as T
from common import CSS, JST, ROOT, nav, write_page

ACTIVE_MINUTES = 10        # 最後に動いてからこの分数以内なら「稼働中」
SHOW_DAYS = 3              # 終わったセッションは、この日数ぶんだけ出す
STATUS_PILL = {"実行中": ("run", "●"), "予約": ("wait", "▲"), "完了": ("done", "○"), "中止": ("stop", "■")}
SURFACE = {"claude-vscode": "VSCode", "sdk-cli": "CLI（-p）", "cli": "端末", "tmux": "tmux"}


def esc(x):
    return html.escape(str(x if x is not None else ""))


def when(ts):
    if not ts:
        return "–"
    d = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
    return d.astimezone(JST).strftime("%m/%d %H:%M")


def ago(d, now):
    m = int((now - d).total_seconds() // 60)
    return "いま" if m < 1 else f"{m} 分前" if m < 60 else f"{m // 60} 時間前" if m < 48 * 60 else f"{m // 1440} 日前"


def pill(cls, mark, label):
    return f'<span class="pill {cls}"><span class="mark">{mark}</span>{esc(label)}</span>'


def tmux_alive(name):
    try:
        return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def task_rows(tasks, now):
    rows = []
    order = {"実行中": 0, "予約": 1, "完了": 2, "中止": 3}
    for t in sorted(tasks, key=lambda t: (order.get(t.get("status"), 9), t["id"])):
        cls, mark = STATUS_PILL.get(t.get("status"), ("done", "○"))
        done, total = A.checklist(t["_dir"])
        pct = round(100 * done / total) if total else 0
        progress = (f'<div class="bar" role="img" aria-label="{done}/{total}"><i style="width:{pct}%"></i></div>'
                    f'<div class="sub">{done} / {total} 項目</div>') if total else '<span class="sub">チェックリストなし</span>'
        place = esc(t.get("host", t.get("created_on", "–")))
        if t.get("surface"):
            place += f'<div class="sub">{esc(t["surface"])}'
            if t.get("tmux"):
                place += f'：{esc(t["tmux"])}' + ("（稼働中）" if tmux_alive(t["tmux"]) else "（終了）")
            place += "</div>"
        reserve = "".join(f'<div class="mono">{esc(p)}/</div>' for p in t.get("paths", []))
        if t.get("ids"):
            reserve += f'<div class="sub">番号：{esc("、".join(t["ids"]))}</div>'
        spec = A.rel(os.path.join(t["_dir"], "指示書.md"))
        bgp = os.path.join(t["_dir"], "経緯.md")
        bg = ("経緯：記入済み" if os.path.exists(bgp) and "<!-- 未記入 -->" not in open(bgp, encoding="utf-8").read()
              else "経緯：未記入（起動できません）" if os.path.exists(bgp) else "経緯：なし")
        rows.append(
            f'<tr><td class="nowrap"><b>{esc(t["id"])}</b></td><td>{pill(cls, mark, t.get("status", ""))}</td>'
            f'<td><b>{esc(t.get("title"))}</b><div class="sub">{esc(t.get("summary"))}</div><div class="mono sub">{esc(spec)}</div><div class="sub">{esc(bg)}</div></td>'
            f'<td>{place}</td><td class="nowrap">{when(t.get("started_at") or t.get("created_at"))}'
            f'<div class="sub">{"終了 " + when(t["finished_at"]) if t.get("finished_at") else ""}</div></td>'
            f'<td>{reserve or "–"}</td><td>{progress}</td></tr>')
    return "\n".join(rows) or '<tr><td colspan="7">タスクはまだありません。<span class="mono">programs/agentctl.py new</span> で作ります。</td></tr>'


def merge_sessions(now):
    """フックの記録と、会話の記録を、セッション ID で合わせる。"""
    _, tr = T.read_all()
    hooks = {s["session_id"]: s for s in A.sessions()}
    out = []
    for sid in set(tr) | set(hooks):
        t, h = tr.get(sid), hooks.get(sid, {})
        last = max([d for d in (t["last"] if t else None,
                                datetime.fromisoformat(h["last_seen"]) if h.get("last_seen") else None) if d], default=None)
        if last is None or (not h and t and t["n_prompts"] == 0):     # 指示が 1 つも無い、空の会話は出さない
            continue
        ended = bool(h.get("ended_at")) or (h.get("pid") and not pid_alive(h["pid"]) and h.get("host") == A.host())
        minutes = (now - last).total_seconds() / 60
        state = "終了" if ended and minutes > 1 else "稼働中" if minutes <= ACTIVE_MINUTES else "待機" if minutes <= 24 * 60 else "終了"
        if state == "終了" and now - last > timedelta(days=SHOW_DAYS):
            continue
        files = sorted((h.get("files") or {}).items(), key=lambda kv: kv[1]["at"], reverse=True)
        out.append(dict(
            session_id=sid, state=state, last=last, first=t["first"] if t else datetime.fromisoformat(h["started_at"]),
            host=h.get("host") or A.host(), surface=h.get("surface") or SURFACE.get((t or {}).get("entrypoint"), (t or {}).get("entrypoint") or "–"),
            cwd=(t or {}).get("cwd") or h.get("cwd") or "", prompt=(t or {}).get("prompt", ""), n_prompts=(t or {}).get("n_prompts", 0),
            task_id=h.get("task_id"), files=files, tracked=bool(h), warnings=h.get("warnings", []), tool_count=h.get("tool_count")))
    order = {"稼働中": 0, "待機": 1, "終了": 2}
    return sorted(out, key=lambda s: (order[s["state"]], -s["last"].timestamp()))


def session_rows(sessions, now):
    cls = {"稼働中": ("run", "●"), "待機": ("wait", "▲"), "終了": ("done", "○")}
    rows = []
    for s in sessions:
        c, mark = cls[s["state"]]
        prompt = " ".join(s["prompt"].split())
        files = "".join(f"<li>{esc(p)}（{v['count']} 回）</li>" for p, v in s["files"][:5])
        more = f'<li>ほか {len(s["files"]) - 5} 件</li>' if len(s["files"]) > 5 else ""
        edited = f'<ul class="files">{files}{more}</ul>' if files else (
            '<span class="sub">なし</span>' if s["tracked"] else '<span class="sub">記録なし（フックを入れる前に始まった会話）</span>')
        folder = (A.rel(s["cwd"]) if s["cwd"] else "").replace(".", A.REPO.name, 1) if A.rel(s["cwd"] or "") == "." else (A.rel(s["cwd"]) if s["cwd"] else "")
        task = f'<b>{esc(s["task_id"])}</b>' if s["task_id"] else '<span class="sub">なし</span>'
        rows.append(
            f'<tr><td>{pill(c, mark, s["state"])}<div class="sub">{ago(s["last"], now)}</div></td>'
            f'<td>{esc(s["host"])}<div class="sub">{esc(s["surface"])}</div></td>'
            f'<td class="nowrap">{when(s["first"])}<div class="sub">指示 {s["n_prompts"]} 回</div></td>'
            f'<td>{task}</td>'
            f'<td>{esc(prompt[:90])}{"…" if len(prompt) > 90 else ""}<div class="mono sub">{esc(folder)}・{esc(s["session_id"][:8])}</div></td>'
            f'<td>{edited}</td></tr>')
    return "\n".join(rows) or '<tr><td colspan="6">記録がありません</td></tr>'


def main():
    now = datetime.now(JST)
    tasks = A.tasks()
    sessions = merge_sessions(now)
    running = [s for s in sessions if s["state"] == "稼働中"]
    by_host = {}
    for s in running:
        by_host[f'{s["host"]}・{s["surface"]}'] = by_host.get(f'{s["host"]}・{s["surface"]}', 0) + 1
    active_tasks = [t for t in tasks if t.get("status") in A.ACTIVE]
    recent = [(s, w) for s in sessions for w in s["warnings"] if now - datetime.fromisoformat(w["at"]) < timedelta(hours=24)]
    warn = ""
    if recent:
        items = "".join(f'<li>{when(w["at"])}　{esc(s["session_id"][:8])}（{esc(s["surface"])}）：{esc(w["text"])}</li>' for s, w in recent[-8:])
        warn = f'<div class="warnbox" role="alert"><b>▲ 直近 24 時間に、予約の外への編集がありました（{len(recent)} 件）</b><ul class="files">{items}</ul></div>'
    tiles = (
        f'<div class="card tile"><div class="label">稼働中の Claude</div><div class="num">{len(running)}</div>'
        f'<div class="sub">{esc("、".join(f"{k} {v}" for k, v in by_host.items()) or "なし")}</div></div>'
        f'<div class="card tile"><div class="label">有効なタスク（予約・実行中）</div><div class="num">{len(active_tasks)}</div>'
        f'<div class="sub">全 {len(tasks)} 件</div></div>'
        f'<div class="card tile"><div class="label">予約されているフォルダ</div><div class="num">{sum(len(t.get("paths", [])) for t in active_tasks)}</div>'
        f'<div class="sub">番号 {sum(len(t.get("ids", [])) for t in active_tasks)} 件</div></div>')
    page = (TEMPLATE.replace("__CSS__", CSS).replace("__NAV__", nav("agents.html"))
            .replace("__UPDATED__", now.strftime("%Y-%m-%d %H:%M")).replace("__TILES__", tiles).replace("__WARN__", warn)
            .replace("__TASKS__", task_rows(tasks, now)).replace("__SESSIONS__", session_rows(sessions, now))
            .replace("__ACTIVE_MIN__", str(ACTIVE_MINUTES)).replace("__SHOW_DAYS__", str(SHOW_DAYS)))
    write_page(ROOT / "agents.html", page)
    print(f"{now.replace(microsecond=0).isoformat()} build agents.html tasks={len(tasks)} sessions={len(sessions)}")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI エージェント</title>
<style>
__CSS__</style>
</head>
<body>
<main>
  __NAV__
  <h1>AI エージェント</h1>
  <p class="meta">最終更新 __UPDATED__（日本時間）
    <button type="button" id="refresh" hidden>更新</button>
    <span id="refresh-note"></span></p>
  __WARN__

  <h2>いまの状態</h2>
  <div class="tiles">__TILES__</div>

  <h2>タスク（予約）</h2>
  <div class="card scroll">
    <table class="list">
      <thead><tr><th>ID</th><th>状態</th><th>タスク・概要・指示書</th><th>動かす端末</th><th>開始</th><th>予約している場所</th><th>進み具合</th></tr></thead>
      <tbody>
__TASKS__
      </tbody>
    </table>
  </div>

  <h2>動いている Claude</h2>
  <div class="card scroll">
    <table class="list">
      <thead><tr><th>状態</th><th>端末・画面</th><th>開始</th><th>タスク</th><th>最初の指示</th><th>最近編集したファイル</th></tr></thead>
      <tbody>
__SESSIONS__
      </tbody>
    </table>
  </div>

  <ul class="notes">
    <li>タスクは、<span class="mono">programs/agentctl.py new</span> で作る予約です。フォルダと番号を予約し、指示書（要件）を書きます。進み具合は、指示書のチェックリストの印の数です。</li>
    <li>「動いている Claude」は、Claude Code のフックと会話の記録から自動で出ます。最後に動いてから __ACTIVE_MIN__ 分以内が「稼働中」、24 時間以内が「待機」です。終了したものは __SHOW_DAYS__ 日ぶん出ます。</li>
    <li>他のタスクが予約している場所を編集しようとすると、その Claude に警告が出ます（編集は止めません）。警告は、この画面の上にも出ます。</li>
    <li>見えるのは、このサーバーで動いた Claude だけです。別の端末（自宅の PC など）で、別の場所のコピーを触っている Claude は出ません。</li>
  </ul>
</main>
<script>
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
