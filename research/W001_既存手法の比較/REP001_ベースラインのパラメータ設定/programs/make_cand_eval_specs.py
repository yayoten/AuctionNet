"""候補の設定で学習した重み（spec_name = candidate_train）を「試す」ための評価 spec を、手法ごとに作る（REP003.md 6 節・7.6 節）。

    .venv/bin/python <REP003>/programs/make_cand_eval_specs.py [BC IQL ...]

条件は 6 節のとおり：エピソード 4〜7（評価に使わない日）、位置 0, 8, 16, 24, 32, 40、学習の seed 1〜3。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[2]
algos = sys.argv[1:] or ["BC", "IQL", "CQL", "BCQ", "TD3_BC"]
found = {}
for mp in sorted((REPO / "DB" / "models").glob("M*/meta.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("rep") == "REP003" and m.get("spec_name") == "candidate_train" and m.get("status") == "ok":
        found.setdefault(m["algo"], {})[m["seed"]] = f"DB/models/{m['model_id']}"
for algo in algos:
    dirs = found.get(algo, {})
    assert sorted(dirs) == [1, 2, 3], f"{algo}: seed 1〜3 の重みがそろっていない（{sorted(dirs)}）"
    spec = {"work": "W001", "rep": "REP003", "name": f"cand_eval_{algo}",
            "note": "候補の設定を試す（6 節）: 30000 ステップ、学習の seed 1,2,3。評価に使わないエピソード 4〜7、6 位置",
            "player": {"strategy": algo, "kwargs": {"model_dir": dirs[1]}},
            "sweep": {"player.kwargs.model_dir": [dirs[s] for s in (1, 2, 3)]},
            "player_indices": [0, 8, 16, 24, 32, 40], "episodes": [4, 5, 6, 7], "seed": 1}
    p = HERE / "params" / f"cand_eval_{algo}.json"
    p.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(p.name, dirs)
