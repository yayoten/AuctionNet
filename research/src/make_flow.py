#!/usr/bin/env python3
"""research/flow.md を、フォルダの構成と各mdの冒頭から生成する（手で編集しない）。

使い方:
  python3 research/src/make_flow.py

mdの冒頭に `---` で囲んで書ける項目（任意）:
  type: 検証 / 探索        （REPのmd）
  status: 支持 / 不支持 / 判断不能 / 未着手 / 検証中 / 達成 / 未達成 / 保留 / 進行中 / 答えが出た
  summary: 1行の要約
"""
import argparse
import re
from datetime import date
from pathlib import Path

from research_tree import RESEARCH_ROOT, scan, walk


def safe(node_id):
    return re.sub(r"\W", "_", node_id)


def label(node):
    kind = ("作業" if node.id.startswith("W") else "問い") if node.kind == "question" else node.meta.get("type", "REP")
    status = node.meta.get("status", "")
    text = f"{node.id} {node.title}"
    return text, kind, status


def outline(nodes, depth=0):
    lines = []
    for n in nodes:
        text, kind, status = label(n)
        tail = f"【{kind}】" + (f" {status}" if status else "")
        summary = n.meta.get("summary")
        lines.append(f"{'  ' * depth}- **{text}** {tail}" + (f"  \n{'  ' * depth}  {summary}" if summary else ""))
        lines += outline(n.children, depth + 1)
    return lines


def mermaid(nodes):
    lines = ["graph TD", "  ROOT[研究マスター]"]
    def edges(parent_id, children):
        for n in children:
            text, kind, status = label(n)
            body = text.replace('"', "'") + (f"<br/>{status}" if status else "")
            shape = f'(["{body}"])' if n.kind == "rep" else f'["{body}"]'
            lines.append(f"  {safe(n.id)}{shape}")
            lines.append(f"  {parent_id} --> {safe(n.id)}")
            edges(safe(n.id), n.children)
    edges("ROOT", nodes)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=RESEARCH_ROOT, help="research/ のパス")
    args = ap.parse_args()

    tree = scan(args.root)
    reps = [n for n in walk(tree) if n.kind == "rep"]
    qs = [n for n in walk(tree) if n.kind == "question"]
    out = [
        "# 問いの木",
        "",
        f"自動生成（{date.today().isoformat()}）。手で編集しない。`python3 research/src/make_flow.py` で作り直す。",
        f"問い {len(qs)} 件 / REP（検証・探索）{len(reps)} 件。",
        "",
    ]
    if not tree:
        out += ["（まだ問いもREPもない）", ""]
    else:
        out += ["## 一覧", ""] + outline(tree) + ["", "## 図", "", "```mermaid"] + mermaid(tree) + ["```", ""]
    (args.root / "flow.md").write_text("\n".join(out), encoding="utf-8")
    print(f"{args.root / 'flow.md'} を更新した（問い {len(qs)}、REP {len(reps)}）")


if __name__ == "__main__":
    main()
