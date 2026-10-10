"""確定した設定（research/src/baseline_params.json）の重みを評価する spec を作る（REP.md 3 節）。

    .venv/bin/python <このREP>/programs/make_eval_specs.py

- eval_<手法>_s<seed>.json：5 手法 × 学習の seed 1〜3。48 位置 × エピソード 0〜3（評価の日）。重みは json の model_dirs から引く。
- eval_rule_based.json：ルールベース 3 手法（既定値）。8 手法の比較を、同じ端末・同じコードの版でそろえるためのもの
  （同じ版の run が DB に既にあれば、run_experiment.py がスキップする）。
- eval_bundled_mbrl.json：MOPO・COMBO（学習コードが無い。同梱の重み）。参考。programs/run_bundled_mbrl.py で流す。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
REP = "REP003new"          # DB の rep の値（既存の REP001〜REP005 と重ならない名前。番号の入れ替えは T005）
base = json.loads((REPO / "research" / "src" / "baseline_params.json").read_text(encoding="utf-8"))
common = {"player_indices": list(range(48)), "episodes": base["evaluation"]["episodes_reserved_for_evaluation"], "seed": 1}
assert common["episodes"] == [0, 1, 2, 3]


def write(name, spec):
    p = HERE / "params" / f"{name}.json"
    p.write_text(json.dumps({"work": "W001", "rep": REP, "name": name, **spec, **common}, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    print(p.name)


for algo, e in base["learned"].items():
    for seed in ("1", "2", "3"):
        write(f"eval_{algo}_s{seed}", {
            "note": f"確定した設定（baseline_params.json、{e['train']['step_num']} ステップ、学習の seed {seed}）の重みを、評価の日・48 位置で評価する",
            "player": {"strategy": e["strategy"], "kwargs": {"model_dir": e["model_dirs"][seed]}}})
rb = base["rule_based"]
assert all(rb[s]["kwargs"] == {} for s in ("PID", "ABid", "OnlineLP"))
write("eval_rule_based", {
    "note": "8 手法の比較用：ルールベース 3 手法（baseline_params.json の rule_based。既定値）を、学習ベースと同じ端末・同じコードの版でそろえる",
    "player": {"strategy": "PID", "kwargs": {}}, "sweep": {"player.strategy": ["PID", "ABid", "OnlineLP"]}})
write("eval_bundled_mbrl", {
    "note": "参考：学習コードが無い MOPO・COMBO を、シミュレータに同梱の重みのまま、プレイヤーとして評価する（確定した設定の重みではない）",
    "player": {"strategy": "MOPO", "kwargs": {}}, "sweep": {"player.strategy": ["MOPO", "COMBO"]}})
