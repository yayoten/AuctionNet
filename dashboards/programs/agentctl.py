#!/usr/bin/env python3
"""エージェントのタスク（予約）を作り、tmux の中で Claude を起動し、終わらせる。

    agentctl.py new --title 題 [--summary 概要] [--paths フォルダ ...] [--ids REP006 ...]   予約を作る（指示書.md のひな形つき）
    agentctl.py list                                                                    一覧
    agentctl.py launch T001 [--model opus]                                              tmux の中で Claude を起動する（確認なしで実行）
    agentctl.py done T001 / cancel T001                                                 完了／中止にする（予約を外す）
    agentctl.py stop T001                                                               tmux を止める（状態は「中止」）

- 予約するのは、フォルダ（--paths。リポジトリ直下からの相対パス）と番号（--ids）。ファイル 1 つずつは予約しない。
- 起動した Claude は、ssh や Wi-Fi が切れても動き続ける。様子を見るには `tmux attach -t agent-T001`（離れるのは Ctrl-b → d）。
- 起動は `claude --dangerously-skip-permissions`（確認なしで全部実行）。守ることは .claude/rules/unattended.md に書いてある。
"""
import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents_lib as A  # noqa: E402

SPEC_TEMPLATE = """# {id} {title}

> 無人で動く Claude への指示書。起動した Claude は、まずこの文書と `.claude/rules/unattended.md` を読む。
> 進み具合は、下のチェックリストに印を付け（`- [x]`）、`進捗.md` に書く。画面の「進み具合」は、この印の数で出る。

## 目的

{summary}

## やること（チェックリスト）

- [ ] （1 つ目の作業。終わったら [x] にする）
- [ ] （2 つ目の作業）

## 終わりの条件

（何ができていたら終わりか）

## 触ってよい場所（予約）

{paths}

## やってはいけないこと

`.claude/rules/unattended.md` のとおり。このタスクに固有のものがあれば、ここに足す。
"""


def tmux_name(task_id):
    return f"agent-{task_id}"


def tmux_alive(name):
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def load(task_id):
    p = A.TASKS / task_id / "task.json"
    t = A.read_json(p)
    if not t:
        sys.exit(f"タスク {task_id} がありません（{p}）")
    return p, t


def cmd_new(a):
    tid = A.next_task_id()
    d = A.TASKS / tid
    paths = [str(Path(p)).rstrip("/") for p in a.paths]
    busy = [(t["id"], f) for t in A.tasks() if t.get("status") in A.ACTIVE for f in t.get("paths", [])
            for p in paths if A.under(A.REPO / p, f) or A.under(A.REPO / f, p)]
    ids_busy = [(t["id"], i) for t in A.tasks() if t.get("status") in A.ACTIVE for i in t.get("ids", []) if i in a.ids]
    if (busy or ids_busy) and not a.force:
        sys.exit(f"予約が重なっています：{busy + ids_busy}。重ねてよければ --force を付けてください。")
    A.write_json(d / "task.json", {"id": tid, "title": a.title, "summary": a.summary, "status": "予約", "paths": paths, "ids": a.ids,
                                   "created_at": A.now(), "created_on": A.host(), "session_ids": []})
    (d / "指示書.md").write_text(SPEC_TEMPLATE.format(id=tid, title=a.title, summary=a.summary or "（何のためにやるか）",
                                                   paths="\n".join(f"- `{p}/`" for p in paths) or "（予約なし）"), encoding="utf-8")
    (d / "進捗.md").write_text(f"# {tid} の進捗\n\n（Claude が、区切りごとに日時つきで足す）\n", encoding="utf-8")
    print(f"{tid} を作りました。指示書を書いてください：{A.rel(d / '指示書.md')}")


def cmd_list(a):
    for t in A.tasks():
        done, total = A.checklist(t["_dir"])
        alive = "tmux 稼働中" if t.get("tmux") and tmux_alive(t["tmux"]) else ""
        print(f"{t['id']}  {t.get('status', ''):4}  {done}/{total}  {t.get('title', '')}  {','.join(t.get('paths', []))}  {alive}")


def cmd_launch(a):
    p, t = load(a.task)
    if not shutil.which("tmux") or not shutil.which("claude"):
        sys.exit("tmux または claude が見つかりません")
    name = tmux_name(a.task)
    if tmux_alive(name):
        sys.exit(f"もう動いています。様子を見るには：tmux attach -t {name}")
    spec = A.rel(A.TASKS / a.task / "指示書.md")
    prompt = (f"あなたは、無人で動くエージェントです。タスク {a.task} を担当します。まず .claude/rules/unattended.md と {spec} を読み、"
              f"指示書のとおりに最後まで進めてください。区切りごとに、指示書のチェックリストに印を付け、同じフォルダの 進捗.md に書いてください。"
              f"終わったら python3 dashboards/programs/agentctl.py done {a.task} を実行してください。")
    claude = ["claude", "--dangerously-skip-permissions"] + (["--model", a.model] if a.model else []) + [prompt]
    inner = f"export AGENT_TASK={shlex.quote(a.task)}; cd {shlex.quote(str(A.REPO))} && {' '.join(shlex.quote(c) for c in claude)}"
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "-x", "200", "-y", "50", inner], check=True)
    t.update(status="実行中", started_at=A.now(), tmux=name, host=A.host(), surface="tmux")
    t.pop("_dir", None)
    A.write_json(p, t)
    print(f"{a.task} を起動しました。様子を見る：tmux attach -t {name}（離れるのは Ctrl-b → d）")


def finish(a, status):
    p, t = load(a.task)
    t.update(status=status, finished_at=A.now())
    t.pop("_dir", None)
    A.write_json(p, t)
    print(f"{a.task} を「{status}」にしました（予約は外れます）")


def cmd_stop(a):
    name = tmux_name(a.task)
    if tmux_alive(name):
        subprocess.run(["tmux", "kill-session", "-t", name], check=True)
    finish(a, "中止")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--title", required=True)
    n.add_argument("--summary", default="")
    n.add_argument("--paths", nargs="*", default=[])
    n.add_argument("--ids", nargs="*", default=[])
    n.add_argument("--force", action="store_true")
    sub.add_parser("list")
    la = sub.add_parser("launch")
    la.add_argument("task")
    la.add_argument("--model")
    for name in ("done", "cancel", "stop"):
        sub.add_parser(name).add_argument("task")
    a = ap.parse_args()
    {"new": cmd_new, "list": cmd_list, "launch": cmd_launch, "stop": cmd_stop,
     "done": lambda a: finish(a, "完了"), "cancel": lambda a: finish(a, "中止")}[a.cmd](a)


if __name__ == "__main__":
    main()
