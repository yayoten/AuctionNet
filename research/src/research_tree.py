"""research/ のフォルダ構成を読む共通部分（build_summary.py と make_flow.py が使う）。

フォルダ名の規則（research/README.md「4. IDの付け方」「5. ディレクトリ構成」）：
  問い    {Q001|Q001-A|Q001-A1|Q001-A1a}_{題}/   （同名の md を置く：{ID}.md）
  作業    W001_{題}/                              （問いが立つ前のまとまり。Q と同じ粒度で、問いと同じ扱い）
  REP     REP001_{題}/                            （REP001.md, REP001.tex, REP001.pdf）
mdの冒頭に `---` で囲んだ `key: value` があれば読む（任意）。使うキーは type / status / summary。
"""
import re
from pathlib import Path

QUESTION_RE = re.compile(r"^(Q\d{3}(?:-[A-Z](?:\d+(?:[a-z]\d*)*)?)?|W\d{3})_(.+)$")  # W001: 問いが立つ前の「作業」。Q と同じ粒度
REP_RE = re.compile(r"^(REP\d{3})_(.+)$")

RESEARCH_ROOT = Path(__file__).resolve().parent.parent


def read_front_matter(md_path):
    """md の冒頭の `---` で囲まれた `key: value` を dict で返す。なければ空。"""
    if not md_path.is_file():
        return {}
    lines = md_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return meta
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.split("#")[0].strip()
    return {}


class Node:
    def __init__(self, path, kind, node_id, title):
        self.path, self.kind, self.id, self.title = path, kind, node_id, title
        self.children = []
        self.meta = read_front_matter(path / f"{node_id}.md")

    @property
    def pdf(self):
        return self.path / f"{self.id}.pdf"

    @property
    def tex(self):
        return self.path / f"{self.id}.tex"


def scan(folder):
    """folder 直下の問い・REP を Node のリストで返す（ID の昇順）。再帰的に子を持つ。"""
    nodes = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        m = QUESTION_RE.match(child.name)
        kind = "question"
        if not m:
            m = REP_RE.match(child.name)
            kind = "rep"
        if not m:
            continue
        node = Node(child, kind, m.group(1), m.group(2))
        if kind == "question":
            node.children = scan(child)
        nodes.append(node)
    nodes.sort(key=lambda n: (n.kind != "question", n.id))
    return nodes


def walk(nodes):
    for n in nodes:
        yield n
        yield from walk(n.children)


def find_question(root, ident):
    """ID（例 Q001-A）またはフォルダのパスから、問いの Node を探す。"""
    p = Path(ident)
    if p.is_dir():
        m = QUESTION_RE.match(p.resolve().name)
        if m:
            return Node(p.resolve(), "question", m.group(1), m.group(2)), True
    for n in walk(scan(root)):
        if n.kind == "question" and n.id == ident:
            return n, False
    return None, False
