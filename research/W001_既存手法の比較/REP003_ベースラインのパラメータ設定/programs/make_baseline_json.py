"""確定したベースラインの設定を、research/src/baseline_params.json に書き出す（REP003.md 8 節）。

    .venv/bin/python <REP003>/programs/make_baseline_json.py --commit <確定したコミット>

- ルールベース 3 手法：本家の既定値（kwargs は空）。
- 学習ベース 5 手法：spec_name = candidate_train の重み（DB/models/）と、その学習の条件。
- 「試した」結果（results/cand_summary.json）と、落ち着きの判定（results/loop1_summary.json）を、手法ごとに添える。
手で json を書き換えない。変えるときは、このスクリプトを直して作り直す。
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
OUT = REPO / "research" / "src" / "baseline_params.json"
REP_DIR = HERE.relative_to(REPO).as_posix()
ALGOS = ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]
DEFAULT_STEPS = {"BC": 20000, "IQL": 20000, "CQL": 100, "BCQ": 100, "TD3_BC": 100}
CAVEAT = {
    "BC": "S*=30000 は、30000 と 50000 ステップの 1 組の比較だけで決まっている",
    "IQL": "S*=30000 は、30000 と 50000 ステップの 1 組の比較だけで決まっている",
    "CQL": "基準では落ち着かない。出力の水準が学習の長さで動く（プローブの平均の α：10000 で 91、20000 で 101、30000 で 91、50000 で 73）。seed の平均では消えない",
    "BCQ": "1000 ステップ以降、状態を見ずに α=100（max_action）を出す。一定の入札の手法として読む。上限を 300 にしても直らなかった",
    "TD3_BC": "基準では落ち着かない。この実装では Q の項がほとんど効かず、実質は BC。試した結果の幅は 9.99% で、基準 10% の際にある",
}

ap = argparse.ArgumentParser()
ap.add_argument("--commit", required=True, help="設定を確定したコミット（REP003 の結果と判断を入れたコミット）")
ap.add_argument("--date", required=True)
a = ap.parse_args()

loop1 = json.loads((HERE / "results" / "loop1_summary.json").read_text(encoding="utf-8"))["by_algo"]
cand = json.loads((HERE / "results" / "cand_summary.json").read_text(encoding="utf-8"))["by_algo"]
metas = {}
for mp in sorted((REPO / "DB" / "models").glob("M*/meta.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("rep") == "REP003" and m.get("spec_name") == "candidate_train" and m.get("status") == "ok":
        metas.setdefault(m["algo"], {})[m["seed"]] = m

learned = {}
for algo in ALGOS:
    ms = metas[algo]
    assert sorted(ms) == [1, 2, 3], (algo, sorted(ms))
    one = ms[1]
    assert len({(m["step_num"], m.get("threads", 1), json.dumps(m.get("train_kwargs") or {})) for m in ms.values()}) == 1
    c = cand[algo]
    assert c["complete"] and c["pass_i_all_seeds_win"] and c["pass_ii_spread_within_10pct"], (algo, c)
    settled = loop1[algo]["S_star"] is not None and algo != "BCQ"
    learned[algo] = {
        "strategy": algo,
        "train": {"step_num": one["step_num"], "official_default_step_num": DEFAULT_STEPS[algo], "seeds": [1, 2, 3],
                  "threads": one.get("threads", 1), "other": one.get("train_kwargs") or {},
                  "train_data_sha1": one["train_data_sha1"], "github_version": one["github_version"],
                  "spec": f"{REP_DIR}/params/" + ("cand_train_bcq.json" if algo == "BCQ" else "cand_train_fast.json")},
        "model_dirs": {str(s): f"DB/models/{ms[s]['model_id']}" for s in (1, 2, 3)},
        "weights_sha1": {str(s): ms[s]["weights_sha1"] for s in (1, 2, 3)},
        "weights_in_git": algo != "BCQ",
        "report": "seed 1〜3 の平均で報告する",
        "settled": settled,
        "S_star": loop1[algo]["S_star"],
        "caveat": CAVEAT[algo],
        "tried": {"episodes": [4, 5, 6, 7], "player_indices": [0, 8, 16, 24, 32, 40], "score_mean_by_seed": c["score_mean_by_seed"],
                  "spread": c["spread"], "passed": True},
    }

out = {
    "version": 1, "decided_in": "REP003", "decided_at": a.date, "commit": a.commit,
    "note": "以後の比較に使う、既存 8 手法のベースライン設定。決め方は REP003.md（research/W001_既存手法の比較/REP003_ベースラインのパラメータ設定/）。"
            "このファイルは programs/make_baseline_json.py で作る（手で書き換えない）",
    "how_to_use": "評価の spec（run_experiment.py）の player に、strategy と kwargs を入れる。学習ベースは kwargs.model_dir に model_dirs の 3 つを順に入れ、3 seed の平均で報告する。"
                  "手法どうしを比べる run は、同じ端末・同じコードの版で流す",
    "rule_based": {
        "PID": {"strategy": "PID", "kwargs": {}}, "ABid": {"strategy": "ABid", "kwargs": {}}, "OnlineLP": {"strategy": "OnlineLP", "kwargs": {}},
        "note": "kwargs が空 = 本家の既定値のまま。評価の日で調整していない（PID・ABid には、評価の日で測ると既定より良い値がある。REP001）",
    },
    "learned": learned,
    "learned_note": "weights_in_git が false の重みは、1 つ 50MB を超えるので git に入れていない。train.spec で学習し直すと、同じ重み（weights_sha1）になる。"
                    "settled は、REP003 の「落ち着いた」の基準（プローブへの出力の差が α で 5 以内）を満たしたか。BCQ は一定の出力で基準を満たしたので false にした",
    "evaluation": {"episodes_used_for_selection": [4, 5, 6, 7], "episodes_reserved_for_evaluation": [0, 1, 2, 3],
                   "note": "設定の選択に使った日は 4〜7。評価の日（0〜3）の score は、選択に使っていない"},
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(OUT)
