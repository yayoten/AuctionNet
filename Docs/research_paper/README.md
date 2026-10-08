# Docs/research_paper/ README

論文の置き場所と、論文調査の決まりをまとめる。
**論文を探す・調べる・引用する作業では、最初にこのファイルを読む。**

> 2026-10-08 時点の方針。

## 1. 目的

- 一度調べた論文を、もう一度調べ直さない。調べたものは、ここに溜めて、あとから引けるようにする。
- 人が読む論文と、調査で集めただけの論文を分ける。全部の論文にフォルダを作ると、人が読む対象が埋もれるため。

## 2. 構成

```
Docs/research_paper/
├── README.md               # このファイル
├── papers.json             # 論文の一覧（番号つきフォルダと papers/ の両方を載せる）
├── request_to_user.md      # 入手できなかった論文（ユーザーに入手を依頼するもの）
├── {番号}_{題}/             # 人が内容を確認する論文（PDF、翻訳など）
└── papers/                 # 調査で集めた論文の PDF（本文は未読のものを含む）
    └── {arXiv ID など}_{題}.pdf
```

| 置き場所 | 何を置くか | 誰が作るか |
|---|---|---|
| `{番号}_{題}/` | 人が目で内容を確認したい論文。PDF と、`/translate-paper` の翻訳・要約 | **ユーザーが指示したときだけ**作る。Claude は自分の判断で作らない |
| `papers/` | 調査で取ってきた論文の PDF。調査の結論に使ったか、使わなかったかは問わない | Claude が調査のたびに足す |
| `papers.json` | 上の 2 つの全論文の一覧 | Claude が、論文を足すたびに更新する |
| `request_to_user.md` | PDF を入手できなかった論文 | Claude が足す。ユーザーが入手したら行を消す |

- `papers/` の論文を人が読むことになったら、ユーザーの指示で `{番号}_{題}/` を作り、PDF を移す。番号は、既存の最大値の次。`papers.json` の `status`・`dir`・`pdf`・`translation` を書き換える。
- `papers/` のファイル名は `{arXiv ID}_{題}.pdf`。題は英数字とハイフンだけにし、空白は `_` にする。arXiv に無い論文は、`{第一著者の姓}{年}_{題}.pdf`。

## 3. 論文を探すときの順序

情報が必要になったら、次の順で探す。前の段で足りれば、次へ進まない。

1. **番号つきフォルダ**（`papers.json` の `status: "read"`）。翻訳があるので、本文まで引ける。
2. **`papers.json` の `status: "pdf_only"`**。`title`・`summary_ja`・`keywords`・`abstract` から近いものを探し、`papers/` の PDF を読んで深掘りする。
3. **Web 検索**（4 節）。

## 4. 論文調査（Web 検索）の決まり

- **1 回の調査につき、Web 検索を 5 回以上行う。** 言い回しや切り口を変えて、幅を取る。
- 見つけた論文は、**PDF を `papers/` に保存する**。調査の結論に使わなかった論文も保存する。
- 保存したら、**その場で `papers.json` に足す**（5 節）。すでに載っている論文は、足さない。
- 足すときに、**PDF の全文で「AuctionNet」を検索し、`auctionnet` の印を付ける**。abstract に無くても、実験の節で使っていることがある。
- 題・著者・年・概要は、論文のページ（arXiv の abstract ページなど）から取る。記憶で書かない。確認できなかった項目は「未確認」と書く。
- **PDF を入手できなかった論文は、`request_to_user.md` に足す**（6 節）。有名・有力な論文ほど、有料で取れないことがある。ユーザーは大学経由で入手できる。
- arXiv は、短時間に続けて取りに行くと拒否される（HTTP 429）。3 秒ほど間隔を空ける。API（`export.arxiv.org`）が拒否されるときは、abstract ページ（`arxiv.org/abs/{ID}`）の `citation_*` のメタ情報を読む。

## 5. `papers.json` の書き方

`papers` の配列に、1 論文 1 件で書く。

| キー | 中身 |
|---|---|
| `key` | 一意な名前。番号つきは `No{番号}`、それ以外は arXiv ID（無ければファイル名の先頭部分） |
| `status` | `read`（番号つきフォルダ。人が読む対象）／ `pdf_only`（`papers/` に PDF だけある。本文は未読） |
| `auctionnet` | 本文に「AuctionNet」の語があるか（`true` / `false`）。同じベンチマークで比較検討できる論文かどうかを見分けるための印。`false` の論文を除くためのものではない |
| `auctionnet_mentions` | 本文中の「AuctionNet」の出現回数（PDF の全文を `pdftotext` で取り出して数える。大文字と小文字は区別しない） |
| `auctionnet_note` | `true` のとき、どう使っているか（実験に使用／言及のみ、など）。`false` のときは `null` |
| `title` | 原題 |
| `authors` | 著者の配列 |
| `year` | 出版年（arXiv のみのときは初版の年） |
| `venue` | 出典（会議・論文誌）。確認できなかったときは、その旨を書く |
| `arxiv_id`、`url` | あれば |
| `dir` | 番号つきフォルダの名前（無ければ `null`） |
| `pdf` | PDF のパス（このフォルダからの相対） |
| `translation` | 翻訳のパス（無ければ `null`） |
| `summary_ja` | 日本語の短い要約。**何を問題にしているか**が分かるように書く。`pdf_only` は abstract だけから書く（本文を読んで書いたものと区別する） |
| `keywords` | 検索用のキーワード（Claude が付ける。英語） |
| `abstract`、`abstract_lang` | 概要の原文（`en`）、または翻訳の Abstract（`ja`） |
| `added` | 足した日 |
| `found_for` | どの調査で集めたか |
| `topic` | 調査での分類（任意） |

- `pdf_only` の論文の PDF を読んで分かったことを足すときは、`notes` に、読んだ範囲と日付つきで書く。`summary_ja` を本文に基づいて書き直したら、`summary_basis: "fulltext"` を足す。
- 一覧を引く例：`python3 -c "import json;[print(p['key'],p['title']) for p in json.load(open('Docs/research_paper/papers.json'))['papers'] if 'pacing' in ' '.join(p['keywords']).lower()]"`

## 6. `request_to_user.md` の書き方

入手できなかった論文を、表の 1 行で書く。列は **題名／掲載誌（会議）／発行年** の 3 つ。

- ユーザーが PDF を入手して置いたら、Claude が `papers/`（または番号つきフォルダ）に移して `papers.json` に足し、表から行を消す。

## 7. 関連

| ファイル | 内容 |
|---|---|
| `.claude/commands/translate-paper.md` | 論文の翻訳・要約（`/translate-paper`） |
| `CLAUDE.md` | 論文調査のときに、最初にこの README を読む、という指示 |
