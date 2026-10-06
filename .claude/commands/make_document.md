# make_document

情報をまとめた資料（解説、調査、説明資料など）を、LaTeX で作って PDF にする。汎用で、型は決めない。
進捗報告（REP）は型があるので `/make_progress_report` を使う。ここでは、体裁だけを共通にして、構成は読み手と目的に合わせて決める。

## 使い方

```
/make_document [題材のメモ、素材のパス、出力先]
```

例：
```
/make_document MTG/2026_10_12/コンペティション解説/ActionNet.pdf  Docs/research_paper/0_.../翻訳.md
```

| ファイル | 役割 |
|---|---|
| `.claude/templates/document/doc_format.sty` | 体裁（`progress_report.sty` と同じ見た目で、サマリーの枠がない）。`\doctitle{}` `\docinfo{}` `\maketitle` |
| `.claude/templates/document/文書テンプレート.tex` | 最小の雛形（2段組み、見出し、図・表の書き方） |

---

## 実行手順

### Step 0: 読み手と目的の確認

次が分からなければ、作り話で埋めず、`AskUserQuestion` か会話で確認する。

- **誰が読むか**（前提知識。例：分野を知らない教授）と、**何のための資料か**（例：進捗の前提を伝える）。
- **ページ数の上限**（目安）。**出力先**（フォルダとファイル名）。
- 載せる内容の指定（含める章、含めない内容）。

読み手の前提に合わせて、用語は初出で説明し、専門外の人が飛ばない書き方にする。

### Step 1: 材料を読む

素材（論文、既存のmd、結果）を、**全文**読む。要約や記憶で書かない。
分からないこと、素材にないことは、推測で書かない。確認できなければ書かないか、「確認していない」と書く。

### Step 2: 構成を決める

読み手と目的から、章立てとページ配分を決める。型はない。迷ったら、「はじめに／本論／まとめ（目的）」の流れにする。
論文を要約するときは、論文の記述と、自分（や実装）から分かったことを、文で区別する（例：「論文の記述」「公開コードの実装では」）。

### Step 3: .tex を作る

1. `.claude/templates/document/文書テンプレート.tex` を、出力先に `{名前}.tex` としてコピーする。使わない部分は消す。
2. `\doctitle{}` にタイトル、`\docinfo{}` に日付などの1行（省略可）を書く。**サマリーの枠はない。**
3. 図は `figs/` に置く。他の資料から使う図は、出典（図の番号と文献）を、キャプションに書く。
4. 2段組みで、横に長い図や、小さい図を多く含む図は、`figure*`（2段にまたがる）の `[t]` にする。1段のままだと文字が読めない。
5. 引用文献は、**原典で確認してから**書く（番号だけから推測しない）。

### Step 4: コンパイル

`.tex` のあるフォルダで、次を実行する。`doc_format.sty` はコピーせず、`TEXINPUTS` で参照する。
`ujarticle` は upLaTeX 前提なので、`pdflatex` は使わない。

```bash
# macOS / Linux
export TEXINPUTS="$(git rev-parse --show-toplevel)/.claude/templates/document//:"
# Windows（Git Bash + MiKTeX）。Windows 形式のパス、区切りはセミコロン
export TEXINPUTS="$(cygpath -m "$(git rev-parse --show-toplevel)")/.claude/templates/document//;"

uplatex -interaction=nonstopmode {名前}.tex
uplatex -interaction=nonstopmode {名前}.tex
dvipdfmx {名前}.dvi
rm -f *.aux *.dvi *.log
```

エラーや Overfull の警告が出たら直す。

### Step 5: 確認と報告

1. PDF を**ページごとに画像にして、目で見る**。日本語の表示、図が小さすぎないか、表のはみ出し、ページ数。
   Windows では poppler（`pdftoppm`）が日本語を描画できないので、PyMuPDF を使う。
   ```python
   import fitz
   d = fitz.open("{名前}.pdf")
   for i, p in enumerate(d): p.get_pixmap(dpi=75).save(f"page{i+1}.png")
   ```
2. 内容を、素材と1つずつ照らす（数値、固有名詞、出典、論文の主張と自分の解釈の区別）。
3. 次を報告する。
   - 出力先（`.tex`、PDF）とページ数
   - 読み手をどう想定したか（資料を渡すときの最初の1行に「Written for: …」の形で書く）
   - 素材から分かったこと・ユーザーの指定と食い違った点（あれば）
   - 確認できなかった点、ユーザーに確認したい点
