"""Claude Code が残す会話の記録（~/.claude/projects/**/*.jsonl）を読む。使用量の画面と、エージェントの画面で使う。

1 行 = 1 つの出来事。assistant の行に、その応答の使用量（message.usage）・モデル・時刻が入っている。
同じ応答（message.id）が複数行に分かれて記録されるので、応答ごとに 1 回だけ数える。
この端末に残っている記録だけが対象（別の端末で動かした Claude のぶんは、ここからは見えない）。
"""
import json
from datetime import datetime
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _text(content):
    """user の行の本文（最初の文章）。道具の結果やシステムの差し込みは除く。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                return c.get("text", "")
    return ""


def read_all(since=None):
    """(応答ごとの使用量のリスト, セッションごとの要約の dict) を返す。since（aware な datetime）より前の応答は捨てる。"""
    responses, sessions = {}, {}
    for path in sorted(PROJECTS.rglob("*.jsonl")):
        sub = "subagents" in path.parts
        try:
            f = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                sid, ts = d.get("sessionId"), d.get("timestamp")
                if not sid or not ts:
                    continue
                try:
                    when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
                s = sessions.setdefault(sid, {"session_id": sid, "first": when, "last": when, "cwd": d.get("cwd"), "entrypoint": d.get("entrypoint"),
                                              "prompt": "", "n_prompts": 0, "path": str(path), "models": set()})
                s["first"], s["last"] = min(s["first"], when), max(s["last"], when)
                s["cwd"] = s["cwd"] or d.get("cwd")
                s["entrypoint"] = s["entrypoint"] or d.get("entrypoint")
                kind = d.get("type")
                if kind == "user" and not sub and not d.get("isSidechain") and not d.get("isMeta"):
                    text = _text((d.get("message") or {}).get("content")).strip()
                    if text and not text.startswith("<") and "tool_result" not in line[:200]:
                        s["n_prompts"] += 1
                        s["prompt"] = s["prompt"] or text
                if kind != "assistant":
                    continue
                m = d.get("message") or {}
                u = m.get("usage")
                if not u or m.get("model") in (None, "<synthetic>"):
                    continue
                s["models"].add(m["model"])
                if since and when < since:
                    continue
                key = (m.get("id") or d.get("uuid"), d.get("requestId"))
                responses[key] = {"when": when, "session_id": sid, "model": m["model"], "project": d.get("cwd") or "",
                                  "entrypoint": d.get("entrypoint") or "", "subagent": sub or bool(d.get("isSidechain")),
                                  **{k: int(u.get(k) or 0) for k in TOKEN_KEYS}}
    return list(responses.values()), sessions
