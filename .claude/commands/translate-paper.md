# translate-paper

英語論文のPDFを受け取り、**日本語翻訳**または**要約**を行うスキル。
論文翻訳モードと論文要約モードの2つのモードを持つ。

## 使い方

```
/translate-paper <PDFファイルまたは論文フォルダの相対パス>
```

例：
```
/translate-paper Docs/research/research_paper/some_paper.pdf
/translate-paper Docs/research/research_paper/No3_Learning_to_Spend_Model_Predictive_Control_for_Budgeting_under_Non-Stationary_Returns
```

---

## 実行手順

引数として受け取ったパスを `$ARGUMENTS` として処理する。

### Step 0: モードとパスの決定

1. `$ARGUMENTS` にモードが明示されていない場合、`AskUserQuestion` で以下を確認する。

   - 質問: 「どちらのモードを実行しますか？」
   - 選択肢:
     - 論文翻訳モード（推奨）: 英語論文PDFを日本語に翻訳し、ディレクトリ構造を整理する
     - 論文要約モード: 既に翻訳済み（または未翻訳）の論文をテンプレートに沿って要約する

2. `$ARGUMENTS` に有効なパス（PDFファイル or 論文フォルダ）が含まれていない場合、ユーザーに対象のパスを確認する。

3. モードが確定したら、対応するセクション（**論文翻訳モード** または **論文要約モード**）に進む。

---

## 論文翻訳モード

これまで通りの翻訳処理を行う。

### Step 1: 論文情報の取得

以下のPythonスクリプトをBashで実行し、PDFからテキストを抽出して論文情報を把握する。

```python
import fitz  # pymupdf
import sys

path = sys.argv[1]
doc = fitz.open(path)

# 最初の2ページのテキストを出力（タイトル・著者・Abstract確認用）
for i in range(min(2, len(doc))):
    print(f"=== Page {i+1} ===")
    print(doc[i].get_text())

# 全ページのテキストを結合して出力
print("\n=== FULL TEXT ===")
for page in doc:
    print(page.get_text())
```

抽出結果から以下を把握する：
- 論文の英語タイトル
- 著者名
- 出版年
- 出典（ジャーナル名 / arXiv IDなど）
- 章構成（"1. Introduction", "2. Methods" などのパターンを検出）

### Step 2: 論文番号の採番とディレクトリ作成

以下のPythonスクリプトで既存フォルダ数を数えて論文番号を決定する。

```python
import os, re

base = "Docs/research/research_paper"
existing = [d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and re.match(r'No\d+_', d)]
next_num = len(existing) + 1
print(next_num)
```

論文番号が決まったら：
- フォルダ名を `No{番号}_{論文名}` の形式で決定する
  - 論文名はスペースをアンダースコアに置換、特殊文字は除去
  - 長すぎる場合は主要な単語を使って50文字以内に収める
- 以下のディレクトリを作成する：
  ```
  Docs/research/research_paper/No{番号}_{論文名}/
  ```

### Step 3: PDFの移動とリネーム

元のPDFファイルを以下のパスにリネームして移動する：
```
Docs/research/research_paper/No{番号}_{論文名}/No{番号}_論文.pdf
```

元のファイルは削除してよい（移動なので `mv` コマンドを使う）。

### Step 4: 翻訳MDファイルの生成

`.claude/templates/translation_template.md` をテンプレートとして使用し、
`Docs/research/research_paper/No{番号}_{論文名}/No{番号}_翻訳.md` を作成する。

**翻訳ルール：**

1. **意味の塊単位の対訳形式** — 論理的にまとまった段落・概念単位で英語原文を示し、その直後に日本語訳を置く。1文ずつの対訳にしない。
   ```
   Marketing has proven to be an important asset to every business.
   Effective strategies allow firms to reach target customers and build lasting relationships.

   マーケティングはあらゆるビジネスにとって重要な資産であり、効果的な戦略によって
   ターゲット顧客へのリーチと長期的な関係構築が可能になる。
   ```

2. **数式のLaTeX記法** — インライン数式は `$...$`、ブロック数式は `$$...$$` で記述する。
   PDFから抽出されたASCIIの数式表現（`x^2`、`sigma` など）はLaTeX記法に変換する。
   ```
   # 変換前（PDF抽出テキスト）
   C(g+1) = (1 - c_mu) C(g) + c_mu ...

   # 変換後（LaTeX記法）
   $$\mathbf{C}^{(g+1)} = (1 - c_\mu)\mathbf{C}^{(g)} + c_\mu \ldots \tag{15}$$
   ```
   重要な式には元論文の式番号をタグ（`\tag{番号}`）で付与する。

3. **章末サマリー** — 各章の最後に引用ブロック形式で章全体の解説を追加する。
   ```
   > **【第X章 まとめ】**
   > （200〜300字で章の内容・ポイント・本論文における位置づけを解説）
   ```

4. **対象セクション** — Abstract から References 直前まで全章を翻訳する。
   References 自体は翻訳不要。

5. **図表の説明文（Figure caption / Table caption）** も翻訳対象とする。
   表は可能な限りMarkdownのテーブル形式で整形する。

6. **重要概念の強調** — 特に重要なキーワードや概念は **太字** でハイライトする。

**出力ファイル構成（テンプレートに従う）：**

```markdown
# No{番号} {論文名} 翻訳

**原題：** ...
**著者：** ...
**出版年：** ...
**出典：** ...

> **翻訳フォーマット**：英語の原文（意味の塊）→ 日本語訳の順に記載。
> 数式は LaTeX 記法（`$...$` / `$$...$$`）で記述。Markdown Preview Enhanced 等で表示推奨。

---

## Abstract

[英語の段落]

日本語訳

> **【Abstract まとめ】**
> ...

---

## 1. Introduction

[英語の段落]

日本語訳

$$\text{重要な数式} \tag{1}$$

数式の説明（日本語）

> **【第1章 まとめ】**
> ...

---
```

### Step 5: 完了報告

以下の情報を出力する：
- 作成したディレクトリパス
- 移動したPDFのパス
- 作成した翻訳MDのパス
- 論文の章構成一覧

### Step 6: 要約モードへの移行提案

翻訳が完了したら、`AskUserQuestion` で以下を確認する。

- 質問: 「翻訳が完了しました。続けてこの論文の要約を作成しますか？」
- 選択肢:
  - はい（フル要約を作成する）（推奨）
  - はい（要約する範囲を選ぶ）
  - いいえ

「はい」が選択された場合、今作成した `No{番号}_翻訳.md` を入力として **論文要約モード** の Step 2 以降（論文番号・論文名の特定は完了しているので Step 3 から）を実行する。

---

## 論文要約モード

既存の論文（翻訳済みMD、または未翻訳のPDF）をテンプレートに沿って要約する。

### Step 1: 対象論文の特定

`$ARGUMENTS` で受け取ったパスから対象論文を特定する。

- 論文フォルダ（例: `Docs/research/research_paper/No3_.../`）が指定された場合は、その中の `No{番号}_翻訳.md` を優先的に読み込む。翻訳MDが存在しない場合は `No{番号}_論文.pdf` からテキストを抽出する（論文翻訳モード Step 1 と同様のPythonスクリプトを使用）。
- 翻訳MDファイルや論文PDFが直接指定された場合は、そのファイルから論文フォルダと論文番号を特定する。

フォルダ名またはファイル名から `No(\d+)_(.+)` のパターンで論文番号と論文名を抽出する。

### Step 2: 論文内容の把握

翻訳MD（または抽出した原文テキスト）を読み込み、論文全体の内容を把握する。

### Step 3: 要約範囲の選択

`AskUserQuestion` で以下を確認する（デフォルトはフル要約）。

- 質問: 「どの範囲を要約しますか？」
- 選択肢:
  - フル要約（推奨・4項目すべて）: 目的と背景／使用した手法／得られた主な結果／解釈と考察のすべてを作成する
  - 目的と背景のみ
  - 使用した手法のみ
  - 得られた主な結果のみ
  - 解釈と考察のみ

複数の項目を個別に組み合わせたい場合は、ユーザーに自由記述（Other）で対象項目を列挙してもらう。

### Step 4: 出力ディレクトリの作成

以下のディレクトリを作成する：
```
Docs/research/research_paper/No{番号}_{論文名}/No{番号}_要約/
```

### Step 5: 要約ファイルの生成

`.claude/templates/summary_template.md` を参照し、選択された項目ごとに対応するプロンプト（`<research_purpose_and_background>` などのタグ内容）に従って要約文を作成する。

選択された各項目について、以下の命名規則でファイルを作成する：

| 項目 | テンプレート内タグ | 出力ファイル |
|---|---|---|
| 目的と背景 | `research_purpose_and_background` | `No{番号}_1_目的と背景.md` |
| 使用した手法 | `methods_and_comparison` | `No{番号}_2_使用した手法.md` |
| 得られた主な結果 | `main_results_and_details` | `No{番号}_3_主な結果.md` |
| 解釈と考察 | `interpretations_and_discussions` | `No{番号}_4_解釈と考察.md` |

フル要約が選択された場合は4ファイルすべてを作成する。個別項目が選択された場合は該当ファイルのみを作成する。

各ファイルは以下の形式で出力する：

```markdown
# No{番号} {論文名} - {項目名}

（テンプレートの指示に従った要約本文。論文からの引用は「> 」の引用ブロックまたは "..." で明示する）
```

**要約の遵守事項（テンプレート共通のnoteに従う）：**
- 論文の内容に忠実に、論文に書かれていない情報や著者の意図を超えた解釈は避ける。
- 論文から直接引用する場合は引用部分を明示する。
- 各項目とも1000〜1500文字程度でまとめる。
- 専門用語には説明を加える。
- 読み手にわかりやすい論理的な段落構成にする。

### Step 6: 完了報告

以下の情報を出力する：
- 作成した要約ディレクトリパス
- 作成した要約ファイルの一覧
