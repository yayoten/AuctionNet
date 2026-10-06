#!/usr/bin/env python3
"""問いのまとめPDFを作る。配下のすべてのREPのPDFを、REP番号の順に結合する。

使い方:
  python3 research/src/build_summary.py Q001-A          # 問いのID
  python3 research/src/build_summary.py research/Q001_xxx/Q001-A_yyy   # フォルダのパス
  python3 research/src/build_summary.py --all           # すべての問い

出力: {問いのフォルダ}/{ID}_まとめ.pdf（作り直せる生成物。手で編集しない）
REPのPDFは `REP001.pdf` のように、REPのフォルダにあるものを使う。
"""
import argparse
import subprocess
import sys
from pathlib import Path

from research_tree import RESEARCH_ROOT, find_question, scan, walk


def page_count(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return 0


def build(question):
    reps = sorted((n for n in walk(question.children) if n.kind == "rep"), key=lambda n: n.id)
    if not reps:
        print(f"[{question.id}] 配下にREPがない。スキップ")
        return True
    missing = [r.id for r in reps if not r.pdf.is_file()]
    if missing:
        print(f"[{question.id}] PDFがないREP: {', '.join(missing)}。先に各REPをコンパイルすること", file=sys.stderr)
        return False
    for r in reps:
        if r.tex.is_file() and r.tex.stat().st_mtime > r.pdf.stat().st_mtime:
            print(f"[{question.id}] 警告: {r.id}.tex が {r.id}.pdf より新しい（PDFが古い可能性）", file=sys.stderr)
    out = question.path / f"{question.id}_まとめ.pdf"
    subprocess.run(["pdfunite", *[str(r.pdf) for r in reps], str(out)], check=True)
    pages = page_count(out)
    note = "" if pages == len(reps) else f"（REPの数 {len(reps)} と一致しない。1ページを超えるREPがある）"
    print(f"[{question.id}] {out}  REP {len(reps)} 本 / {pages} ページ {note}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="問いのID、または問いのフォルダのパス")
    ap.add_argument("--all", action="store_true", help="すべての問いのまとめを作る")
    ap.add_argument("--root", type=Path, default=RESEARCH_ROOT, help="research/ のパス")
    args = ap.parse_args()

    if args.all:
        ok = all([build(q) for q in walk(scan(args.root)) if q.kind == "question"])
        sys.exit(0 if ok else 1)
    if not args.target:
        ap.error("問いのID・パス、または --all を指定すること")
    question, from_path = find_question(args.root, args.target)
    if question is None:
        print(f"問いが見つからない: {args.target}", file=sys.stderr)
        sys.exit(1)
    if from_path:
        question.children = scan(question.path)
    sys.exit(0 if build(question) else 1)


if __name__ == "__main__":
    main()
