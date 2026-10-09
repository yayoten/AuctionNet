#!/usr/bin/env python3
"""Claude Code のフックから呼ばれ、動いている Claude を台帳（agents/sessions/）に記録する。

    agent_hook.py <start|pre|post|stop|end>     （フックの入力 JSON を標準入力から読む）

- start：台帳に登録する。環境変数 AGENT_TASK があれば、そのタスクに結び付ける。
- pre  ：ファイルを編集する直前。他のタスクが予約している場所、またはホームの外なら、警告を返す（止めはしない）。
- post ：道具を使ったあと。最後に動いた時刻と、編集したファイルを記録する。
- stop ：1 回の応答が終わった。  end：セッションが終わった。
どんな失敗をしても、Claude の作業は止めない（終了コードは常に 0）。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents_lib as A  # noqa: E402

EDIT_TOOLS = ("Edit", "Write", "NotebookEdit")
KEEP_FILES = 30


def load(session_id):
    return A.read_json(A.SESSIONS / f"{session_id}.json") or {"session_id": session_id, "started_at": A.now(), "files": {}, "tool_count": 0}


def save(s):
    s["last_seen"] = A.now()
    A.write_json(A.SESSIONS / f"{s['session_id']}.json", s)


def edited_path(data):
    ti = data.get("tool_input") or {}
    return ti.get("file_path") or ti.get("notebook_path")


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    sid = data.get("session_id")
    if not sid:
        return
    s = load(sid)
    s.update(host=A.host(), surface=s.get("surface") or A.surface(), cwd=data.get("cwd") or s.get("cwd"),
             transcript_path=data.get("transcript_path") or s.get("transcript_path"), last_event=event)
    s.pop("ended_at", None) if event != "end" else None

    if event == "start":
        s["pid"] = os.getppid()
        task = os.environ.get("AGENT_TASK")
        if task:
            s["task_id"] = task
            tp = A.TASKS / task / "task.json"
            t = A.read_json(tp)
            if t:
                if sid not in t.setdefault("session_ids", []):
                    t["session_ids"].append(sid)
                if t.get("status") == "予約":
                    t.update(status="実行中", started_at=A.now())
                t.update(host=A.host(), surface=s["surface"])
                A.write_json(tp, t)
        save(s)
        return

    if event == "pre":
        path = edited_path(data)
        notes = []
        if path:
            if A.outside_home(path):
                notes.append(f"【台帳からの警告】{path} は、ホーム（{A.HOME}）の外です。ホームの外は触らない決まりです。")
            for t in A.conflicts(path, sid, s.get("task_id")):
                notes.append(f"【台帳からの警告】{A.rel(path)} は、タスク {t['id']}「{t.get('title', '')}」（{t.get('status')}）が予約している場所です。"
                             f"そのタスクの担当でなければ、編集する前に dashboards/agents/tasks/{t['id']}/ を確認してください。")
        if notes:
            s.setdefault("warnings", []).append({"at": A.now(), "path": A.rel(path), "text": notes[0]})
            s["warnings"] = s["warnings"][-20:]
            save(s)
            text = "\n".join(notes)
            print(json.dumps({"systemMessage": text,
                              "hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": text}}, ensure_ascii=False))
        return

    if event == "post":
        s["tool_count"] = s.get("tool_count", 0) + 1
        s["last_tool"] = data.get("tool_name")
        path = edited_path(data) if data.get("tool_name") in EDIT_TOOLS else None
        if path:
            r = A.rel(path)
            files = s.setdefault("files", {})
            files[r] = {"count": files.get(r, {}).get("count", 0) + 1, "at": A.now()}
            if len(files) > KEEP_FILES:      # 新しいものだけ残す
                for k in sorted(files, key=lambda k: files[k]["at"])[:-KEEP_FILES]:
                    del files[k]
        save(s)
        return

    if event == "end":
        s["ended_at"] = A.now()
    save(s)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # フックの失敗で Claude を止めない
        pass
    sys.exit(0)
