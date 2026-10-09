"""エージェントの台帳（dashboards/agents/）の読み書き。フック・agentctl・画面づくりで共通に使う。

台帳は 2 種類のファイルからなる。1 件 1 ファイルにして、台帳そのものが取り合いにならないようにする。
- agents/tasks/<ID>/task.json … タスク（予約）。人か Claude が agentctl.py で作る。指示書.md と 進捗.md が並ぶ。git で管理する。
- agents/sessions/<session_id>.json … 動いている Claude 1 つ。フックが自動で書く。git では管理しない。
"""
import json
import os
import re
import socket
from datetime import datetime
from pathlib import Path

from common import JST, REPO, ROOT

AGENTS = ROOT / "agents"
TASKS = AGENTS / "tasks"
SESSIONS = AGENTS / "sessions"
HOME = Path.home()
ACTIVE = ("予約", "実行中")              # この状態のタスクの予約が、警告の対象になる
STATUSES = ("予約", "実行中", "完了", "中止")


def now():
    return datetime.now(JST).replace(microsecond=0).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, data):
    """途中まで書いたファイルを他の読み手に見せないよう、一時ファイルに書いてから置き換える。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def host():
    return socket.gethostname()


def surface():
    """この Claude が、どの画面から動かされているか（tmux の中か、VSCode か、ただの端末か）。"""
    if os.environ.get("TMUX"):
        return "tmux"
    if os.environ.get("VSCODE_PID") or os.environ.get("TERM_PROGRAM") == "vscode" or os.environ.get("CLAUDE_CODE_ENTRYPOINT") == "claude-vscode":
        return "VSCode"
    return os.environ.get("CLAUDE_CODE_ENTRYPOINT") or "端末"


def tasks():
    out = []
    for p in sorted(TASKS.glob("*/task.json")):
        t = read_json(p)
        if t:
            t["_dir"] = str(p.parent)
            out.append(t)
    return out


def sessions():
    return [s for s in (read_json(p) for p in sorted(SESSIONS.glob("*.json"))) if s]


def next_task_id():
    nums = [int(m.group(1)) for p in TASKS.glob("T*") if (m := re.fullmatch(r"T(\d+)", p.name))]
    return f"T{max(nums, default=0) + 1:03d}"


def rel(path):
    """リポジトリ直下からの相対パス（外のファイルは絶対パスのまま）。"""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def under(path, folder):
    """path が folder の中（または同じ）か。どちらもリポジトリ直下からの相対パスで比べる。"""
    a, b = rel(path).rstrip("/"), str(folder).rstrip("/")
    return a == b or a.startswith(b + "/")


def checklist(task_dir):
    """指示書.md のチェックリスト（- [ ] / - [x]）の、(済んだ数, 全体の数)。"""
    try:
        text = (Path(task_dir) / "指示書.md").read_text(encoding="utf-8")
    except OSError:
        return 0, 0
    marks = re.findall(r"^\s*[-*]\s*\[([ xX])\]", text, flags=re.M)
    return sum(m in "xX" for m in marks), len(marks)


def conflicts(path, session_id, my_task=None):
    """path を予約している、自分以外の有効なタスク。"""
    out = []
    for t in tasks():
        if t.get("status") not in ACTIVE or t["id"] == my_task or session_id in t.get("session_ids", []):
            continue
        if any(under(path, f) for f in t.get("paths", [])):
            out.append(t)
    return out


def outside_home(path):
    try:
        Path(path).resolve().relative_to(HOME)
        return False
    except ValueError:
        return True
