# git push の決まり（README）

**push してよいのは、AuctionNet のリポジトリの `main` だけ。** それ以外には、何があっても push しない。
その範囲の中では、ユーザーの確認を取らずに、自由に push してよい（無人で動く Claude も同じ）。

> 2026-10-09 作成。ユーザーの決定：「AuctionNet しか push しないという README を作って、push は自由にさせよう」。
> それまでは「push はユーザーが行う（Claude はしない）」だった。この文書が、それを置き換える。

## 1. push してよい場所

| 項目 | 内容 |
|---|---|
| リポジトリ | `https://github.com/yayoten/AuctionNet.git`（この作業フォルダの `origin`） |
| ブランチ | `main` |
| コマンド | `git push origin main` |

**push の直前に、必ず確かめる。** 1 つでも違えば、push しない（無人なら `進捗.md` に書いて止める）。

```bash
git remote get-url origin      # → https://github.com/yayoten/AuctionNet.git であること
git branch --show-current      # → main であること
git fetch origin && git status -sb | head -1     # → ahead だけで、behind が付いていないこと
```

## 2. 絶対にやらないこと

| やらないこと | 理由 |
|---|---|
| AuctionNet 以外のリポジトリ・remote への push（`git remote add` で別の宛先を足すことも含む） | この許可は、AuctionNet だけに出している |
| `main` 以外のブランチ・タグの push | 範囲を絞るため |
| `--force`、`--force-with-lease`、`+main` のような強制 push、リモートの枝・タグの削除 | リモートの履歴は、壊すと戻せない |
| **push 済みの**コミットの書き換え（`reset`、`rebase`、`filter-branch`、`commit --amend`） | 他の場所のコピーと食い違う。**まだ push していないコミット**なら、書き換えてよい（その前に、バックアップの枝を作る） |
| 認証情報（トークン、鍵）を、画面・ログ・コミット・メモに出す、コミットに入れる | 漏れると、リポジトリを書き換えられる |
| 他の Claude の未コミットの変更を、巻き込んでコミットして push する | コミットは、自分が変えたファイルを名前で指定する（`CLAUDE.md`） |

## 3. push の前に確かめること

1. **大きなファイルが入っていないか。** GitHub は、1 ファイル 100MB を超えると拒否する（50MB 超は警告）。
   ```bash
   git rev-list origin/main..HEAD --objects | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
     | awk '$1=="blob" && $2>50000000' | sort -k2 -n -r | head
   ```
   何か出たら、push しない。学習した重みの途中のもの（`DB/models/*/ckpt/*/*.pth`）などは、`.gitignore` で外してある。
2. **秘密が入っていないか。** トークン、鍵、パスワード（`.env`、`*.pem`、`id_*`、`.git-credentials`）が、コミットに入っていないこと。
3. **テストが通っていること**（`github/` を変えたコミットを含むとき。`CLAUDE.md` の決まり）。
4. `behind` が付いていたら（リモートが先に進んでいる）、`git pull --ff-only origin main` を試す。**混ぜる（merge）必要が出たら、止めて報告する。**

## 4. いつ push するか

- 区切りごと（REP の事前記録、結果、報告 PDF、基盤の変更が、コミットできたところ）に push してよい。
- タスクが終わるとき（`agentctl.py done`）の前に、コミットしたものを push する。
- 「サーバーのディスクが壊れても、失われない」ことが、いちばんの目的。たまったままにしない（2026-10-09 時点で、丸 1 日ぶん 23 件がたまっていた）。

## 5. 認証の置き場所

- サーバーには、**この 1 つのリポジトリだけに権限を絞った**トークン（GitHub の Fine-grained token。Repository access は `AuctionNet` のみ、Permissions は `Contents: Read and write` のみ）を、`~/.git-credentials` に置く（`git config --global credential.helper store`）。
- ファイルの権限は 600（自分だけ読める）にする：`chmod 600 ~/.git-credentials`。
- トークンの期限が切れたら、`git push` が認証エラーになる。**Claude が直せることではない。** 無人なら、コミットだけして、`進捗.md` に「push できなかった（認証）」と書く。ユーザーが作り直す。
- トークンの作り方は、ユーザーが GitHub の画面で行う。**トークンの文字列を、Claude との会話に貼らない**（記録に残る）。

## 6. 失敗したとき

- 認証エラー・拒否・`behind`・大きなファイルの検出は、**直そうとして範囲を広げない。** コミットは残したまま、理由を `進捗.md`（無人のとき）か、報告（対話のとき）に書いて、止める。
- 「force すれば通る」と考えたら、それは、やってはいけないこと（2 節）に当たる。
