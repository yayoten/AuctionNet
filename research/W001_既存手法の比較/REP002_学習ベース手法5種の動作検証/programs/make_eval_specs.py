"""学習した重み（DB/models/）から、自前の重みを評価する spec（params/eval_trained_*.json）を作る。

    .venv/bin/python research/W001_.../REP002_.../programs/make_eval_specs.py

model_id は学習してから決まるので、評価の spec は学習のあとに、この決まりで機械的に作る（手で選ばない）。
- eval_trained_default_<手法>：既定のステップ数・seed=1 の重み。48 位置 × エピソード 0〜3（L4）
- eval_trained_steps_<手法>：train_steps（20000 ステップ）の重み。6 位置（L5 追加 2）
- eval_trained_seeds_<手法>：train_seeds（seed 2, 3）の重み。6 位置（L5 追加 3）
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[2]
POS6 = [0, 8, 16, 24, 32, 40]

metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "DB" / "models").glob("M*/meta.json"))]
metas = [m for m in metas if m.get("rep") == "REP002" and m["status"] == "ok"]
for old in (HERE / "params").glob("eval_trained_*.json"):
    old.unlink()


def write(kind, algo, models, positions, note):
    dirs = [f"DB/models/{m['model_id']}" for m in sorted(models, key=lambda m: (m["seed"], m["step_num"]))]
    spec = {"work": "W001", "rep": "REP002", "name": f"eval_trained_{kind}_{algo}", "note": note,
            "player": {"strategy": algo, "kwargs": {"model_dir": dirs[0]}},
            "sweep": {"player.kwargs.model_dir": dirs}, "player_indices": positions, "episodes": [0, 1, 2, 3], "seed": 1}
    (HERE / "params" / f"eval_trained_{kind}_{algo}.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(spec["name"], dirs)


for algo in ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]:
    ms = [m for m in metas if m["algo"] == algo]
    default = [m for m in ms if m["spec_name"] == "train_default"]
    assert len(default) == 1, f"{algo}: 既定設定の重みが 1 つではない（{len(default)}）"
    write("default", algo, default, list(range(48)), "L4: 既定設定で自前学習した重み（48位置）")
    steps = [m for m in ms if m["spec_name"] == "train_steps"]
    if steps:
        write("steps", algo, steps, POS6, "L5 追加2: 20000ステップで学習した重み（6位置）")
    seeds = [m for m in ms if m["spec_name"] == "train_seeds"]
    if seeds:
        write("seeds", algo, seeds, POS6, "L5 追加3: seed 2, 3 で学習した重み（6位置）")
