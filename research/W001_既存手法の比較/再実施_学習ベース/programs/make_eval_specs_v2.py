"""流し直し（T007、記録の形式の版 2）の spec を、params/ の spec から作る（params_v2/ に書く。もとの spec は書き換えない）。

    .venv/bin/python <このREP>/programs/make_eval_specs_v2.py

変えるのは、"rep"（ルールベースは rerun_rule_v2、学習ベースは rerun_learned_v2）と、"record": "standard" の明示だけ。
eval_bundled_mbrl（MOPO・COMBO。参考）は、864 run に入らないので作らない。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC, DST = HERE / "params", HERE / "params_v2"
REP_RULE, REP_LEARNED = "rerun_rule_v2", "rerun_learned_v2"
DST.mkdir(exist_ok=True)
for p in sorted(SRC.glob("eval_*.json")):
    if p.stem == "eval_bundled_mbrl":
        continue
    spec = json.loads(p.read_text(encoding="utf-8"))
    assert spec["rep"] == "REP003new"
    spec["rep"] = REP_RULE if p.stem == "eval_rule_based" else REP_LEARNED
    spec["record"] = "standard"
    (DST / p.name).write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(p.name, spec["rep"])
