"""01: Python 環境と依存パッケージが、本家の指定どおりか。"""
import ast
import importlib
import importlib.metadata as md
import importlib.util
import re
import sys
import sysconfig
from pathlib import Path

import pytest
from packaging.version import Version

from conftest import GITHUB_DIR


def _parse_requirements():
    reqs = {}
    for line in (GITHUB_DIR / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([\w.]+)$", line)
        assert m, f"requirements.txt の行が name==version 形式でない: {line!r}"
        reqs[m.group(1)] = m.group(2)
    return reqs


REQS = _parse_requirements()


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_python_is_39():
    """本家 README は Python 3.9.12（torch==1.12.0 が 3.10 以降に無い）。"""
    assert sys.version_info[:2] == (3, 9), f"Python {sys.version.split()[0]} （3.9 系で検証する）"


def test_requirements_file_parses_and_is_nonempty():
    assert len(REQS) >= 7


@pytest.mark.parametrize("name", list(REQS))
def test_pinned_version_installed(name):
    installed = md.version(name)
    assert Version(installed) == Version(REQS[name]), f"{name}: 指定 {REQS[name]} / 実際 {installed}"


def test_gin_and_gin_config_both_installed_and_coexist():
    """requirements は `gin` と `gin_config` の両方を指定する。両者とも top-level は `gin`。"""
    import gin

    assert hasattr(gin, "configurable")
    assert hasattr(gin, "parse_config_files_and_bindings")
    assert hasattr(gin, "parse_config")
    assert hasattr(gin, "clear_config")


def test_torch_basic_ops_and_cpu_only():
    import torch

    x = torch.arange(6, dtype=torch.float).reshape(2, 3)
    assert torch.equal(x @ x.T, torch.tensor([[5.0, 14.0], [14.0, 50.0]]))


def test_torch_jit_roundtrip(tmp_path):
    """学習済みエージェントは torch.jit.load で読む。torch 1.12 で保存→読込が通ること。"""
    import torch

    m = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh())
    p = tmp_path / "m.pth"
    torch.jit.script(m).save(str(p))
    m2 = torch.jit.load(str(p))
    x = torch.ones(3)
    assert torch.allclose(m(x), m2(x))


def test_scipy_truncnorm_vectorized_matches_env_usage():
    """BiddingEnv._generate_values_matrix と同じ呼び方（a,b が行ベクトル、loc/scale が行列）。"""
    import numpy as np
    from scipy.stats import truncnorm

    loc = np.full((5, 4), 0.01)
    scale = np.full((5, 4), 0.001)
    a = -2 * np.ones((1, 4))
    b = 2 * np.ones((1, 4))
    out = truncnorm.rvs(a, b, loc=loc, scale=scale, random_state=np.random.default_rng(1))
    assert out.shape == (5, 4)
    assert np.all(out >= loc - 2 * scale - 1e-12) and np.all(out <= loc + 2 * scale + 1e-12)


def test_func_timeout_works():
    import time

    from func_timeout import FunctionTimedOut, func_set_timeout

    @func_set_timeout(0.2)
    def slow():
        time.sleep(2)

    with pytest.raises(FunctionTimedOut):
        slow()


def test_psutil_memory_info():
    import os

    import psutil

    assert psutil.Process(os.getpid()).memory_info().rss > 0


def test_numpy_default_rng_is_available():
    import numpy as np

    assert np.random.default_rng(1).random() == np.random.default_rng(1).random()


def test_pandas_csv_roundtrip(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    assert pd.read_csv(p).equals(df)


def _third_party_modules_imported_by_github():
    """github/ 配下の .py が import している、site-packages 由来のトップレベルモジュール名。"""
    site_dirs = {Path(p).resolve() for p in (sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"])}
    tops = set()
    for f in GITHUB_DIR.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tops.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                tops.add(node.module.split(".")[0])
    found = set()
    for t in tops:
        try:
            spec = importlib.util.find_spec(t)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
            if spec is not None and spec.submodule_search_locations:
                origin_dirs = [Path(p).resolve() for p in spec.submodule_search_locations]
            else:
                continue
        else:
            origin_dirs = [Path(spec.origin).resolve()]
        if any(sd in o.parents or o == sd for o in origin_dirs for sd in site_dirs):
            found.add(t)
    return found


def _module_to_dist():
    mapping = {}
    for dist in md.distributions():
        tl = dist.read_text("top_level.txt")
        names = tl.split() if tl else []
        if not names and dist.files:
            names = {str(f).split("/")[0] for f in dist.files if "/" in str(f) and not str(f).endswith(".pyc")}
        for n in names:
            mapping.setdefault(n, set()).add(_norm(dist.metadata["Name"]))
    return mapping


def _missing_from(required):
    mapping = _module_to_dist()
    missing = []
    for mod in sorted(_third_party_modules_imported_by_github()):
        dists = mapping.get(mod, {_norm(mod)})
        if not (dists & required):
            missing.append(mod)
    return missing


def test_third_party_import_scan_finds_the_known_packages():
    found = _third_party_modules_imported_by_github()
    assert {"numpy", "torch", "pandas", "gin", "scipy", "psutil", "func_timeout", "einops", "matplotlib"} <= found


def test_requirements_cover_every_third_party_import():
    """修正前は einops が requirements.txt に無かった。"""
    missing = _missing_from({_norm(n) for n in REQS})
    assert not missing, f"requirements.txt に無いのに import されているパッケージ: {missing}"


def test_requirements_check_would_detect_a_missing_package():
    """この検査が本当に欠落を検出できること（einops を抜いた requirements で確認）。"""
    assert _missing_from({_norm(n) for n in REQS} - {"einops"}) == ["einops"]


def test_einops_is_importable_in_this_venv():
    """ModelPvGen が使う einops が入っていること。"""
    from einops import rearrange  # noqa: F401


# ---------- 他端末での再現用ロックファイル ----------
def _lock():
    out = {}
    for line in (Path(__file__).parent / "requirements.lock.txt").read_text().splitlines():
        if "==" in line:
            n, v = line.strip().split("==")
            out[_norm(n)] = v
    return out


def test_lock_file_pins_every_package_in_requirements_to_the_same_version():
    lock = _lock()
    for name, ver in REQS.items():
        assert _norm(name) in lock, name
        assert Version(lock[_norm(name)]) == Version(ver), name


def test_lock_file_includes_test_tools():
    assert {"pytest", "pytest-timeout"} <= set(_lock())


def test_installed_packages_match_lock_file():
    """この環境が、ロックファイルどおりの版で出来ていること（別端末で作った環境の確認にもなる）。"""
    diff = {n: (v, md.version(n)) for n, v in _lock().items() if Version(md.version(n)) != Version(v)}
    assert not diff, f"ロックと違う版: {diff}"
