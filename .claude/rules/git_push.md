# git push の決まり（README）

**push してよいのは、AuctionNet のリポジトリの `main` だけ。** それ以外には、何があっても push しない。
**ただし、push するときは、毎回、先にユーザーの承認を得る。** 承認を得ずに push しない。無人で動く Claude は、承認を得られないので push せず、コミットまでにして、`進捗.md` に「push 待ち」と書く。

> 2026-10-09 作成。ユーザーの決定：「AuctionNet しか push しないという README を作って、push は自由にさせよう」。
> それまでは「push はユーザーが行う（Claude はしない）」だった。この文書が、それを置き換えた。
> **2026-10-10 に、ユーザーの指示で変更：「push するときには、ユーザに承認を得てからにして」。** 宛先の範囲（AuctionNet の `main` だけ）は、そのまま。承認は、push のたびに取る（前の承認は、次の push に及ばない）。

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

### まだ push していないコミットを書き換えるとき

1. バックアップの枝を作る（`git branch backup-<名前> HEAD`）。
2. 作業ツリーに他の Claude の未コミットの変更があると、`git filter-branch` は拒否する。**その変更に触らず**、`git worktree add -b <枝> ~/wt-<名前> HEAD` で別の作業ツリーを作り、そこで書き直す。
3. 差し替えは、`git update-ref refs/heads/main <新> <旧>`（旧の位置を指定するので、その間に `main` が動いていたら失敗する）。そのあと、メインの作業ツリーの索引（index）だけを、`git rm --cached` で合わせる。
4. 書き直した結果の差が、意図したものだけか（`git diff --name-only <旧> <新>`）を確かめる。
5. push が済んだら、バックアップの枝は消してよい（`git branch -D`、`git reflog expire --expire=now --all`、`git gc --prune=now`）。

## 3. push の前に確かめること

1. **大きなファイルが入っていないか。** GitHub は、1 ファイル 100MB を超えると拒否する（50MB 超は警告）。
   ```bash
   git rev-list origin/main..HEAD --objects | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
     | awk '$1=="blob" && $2>50000000' | sort -k2 -n -r | head
   ```
   何か出たら、push しない。学習した重みの途中のもの（`DB/models/*/ckpt/*/*.pth`）などは、`.gitignore` で外してある。
2. **`.gitignore` の行末にコメントを書いていないか。** 行末のコメントは、パターンの一部になり、効かなくなる。コメントは、別の行に書く（2026-10-09、学習の途中の重み 約 1.3GB が、これでコミットに入った）。
3. **秘密が入っていないか。** トークン、鍵、パスワード（`.env`、`*.pem`、`id_*`、`.git-credentials`）が、コミットに入っていないこと。
4. **テストが通っていること**（`github/` を変えたコミットを含むとき。`CLAUDE.md` の決まり）。
5. `behind` が付いていたら（リモートが先に進んでいる）、`git pull --ff-only origin main` を試す。**混ぜる（merge）必要が出たら、止めて報告する。**

## 4. いつ push するか

- 区切りごと（REP の事前記録、結果、報告 PDF、基盤の変更が、コミットできたところ）で、**push してよいかをユーザーに聞く。** 承認が出たら push する。
- 対話のときは、区切りでコミットしたあと、「push してよいか」を一言で聞く（何件・何が入っているかを添える）。
- 無人のときは、聞けないので push しない。タスクが終わるとき（`agentctl.py done`）の前に、コミットまでを済ませ、`進捗.md` に「push 待ち（コミット一覧）」と書く。ユーザーが承認したら、あとから push する。
- 「サーバーのディスクが壊れても、失われない」ことが、いちばんの目的。たまったままにしない（2026-10-09 時点で、丸 1 日ぶん 23 件がたまっていた）。そのため、区切りごとに、忘れずに承認を求める。

## 5. 認証の置き場所

- サーバーには、**この 1 つのリポジトリだけに権限を絞った**トークン（GitHub の Fine-grained token。Repository access は `AuctionNet` のみ、Permissions は `Contents: Read and write` のみ）を、`~/.git-credentials` に置く（`git config --global credential.helper store`）。
- ファイルの権限は 600（自分だけ読める）にする：`chmod 600 ~/.git-credentials`。
- トークンに**期限は付けない**（無期限。ユーザーの決定、2026-10-09）。そのため、**漏れたときは、ユーザーが手で削除するまで、使われ続ける。** トークンは、サーバーの `~/.git-credentials` と、ユーザーの MacBook のキーチェーンにだけ置く。それ以外の場所（リポジトリ、ログ、メモ、会話）に出さない。
- `git push` が認証エラーになったら、トークンが削除された（または権限が変わった）ということ。**Claude が直せることではない。** 無人なら、コミットだけして、`進捗.md` に「push できなかった（認証）」と書く。ユーザーが作り直す。
- トークンの作り方は、ユーザーが GitHub の画面で行う。**トークンの文字列を、Claude との会話に貼らない**（記録に残る）。

## 6. 失敗したとき

- 認証エラー・拒否・`behind`・大きなファイルの検出は、**直そうとして範囲を広げない。** コミットは残したまま、理由を `進捗.md`（無人のとき）か、報告（対話のとき）に書いて、止める。
- 「force すれば通る」と考えたら、それは、やってはいけないこと（2 節）に当たる。
