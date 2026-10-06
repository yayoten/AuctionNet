"""03: 本家の全モジュールが import できるか（副作用の有無も確認）。"""
import importlib
import os
import subprocess
import sys

import pytest

from conftest import GITHUB_DIR, REPO_ROOT


def _all_modules():
    mods = []
    for f in sorted(GITHUB_DIR.rglob("*.py")):
        if f.name == "__init__.py":
            continue
        rel = f.relative_to(REPO_ROOT).with_suffix("")
        mods.append(".".join(rel.parts))
    return mods


ALL_MODULES = _all_modules()


def test_module_discovery_found_expected_count():
    assert len(ALL_MODULES) >= 60


@pytest.mark.parametrize("mod", ALL_MODULES)
def test_module_imports(mod):
    assert importlib.import_module(mod) is not None


@pytest.mark.parametrize("pkg", [
    "github.simul_bidding_env", "github.simul_bidding_env.Controller", "github.simul_bidding_env.Environment",
    "github.simul_bidding_env.PvGenerator", "github.simul_bidding_env.Tracker", "github.simul_bidding_env.strategy",
    "github.run", "github.strategy_train_env.bidding_train_env",
    "github.strategy_train_env.bidding_train_env.baseline", "github.strategy_train_env.bidding_train_env.common",
    "github.strategy_train_env.bidding_train_env.offline_eval", "github.strategy_train_env.bidding_train_env.strategy",
    "github.strategy_train_env.bidding_train_env.train_data_generator",
])
def test_package_imports(pkg):
    assert importlib.import_module(pkg) is not None


CLEAN_IMPORT_TARGETS = [
    "github.run.run_test",
    "github.simul_bidding_env.Controller.Controller",
    "github.simul_bidding_env.Environment.BiddingEnv",
    "github.simul_bidding_env.PvGenerator.NeurIPSPvGen",
    "github.simul_bidding_env.Tracker.BiddingTracker",
    "github.simul_bidding_env.Tracker.PlayerAnalysis",
    "github.strategy_train_env.bidding_train_env.train_data_generator.train_data_generator",
]


@pytest.mark.slow
@pytest.mark.parametrize("mod", CLEAN_IMPORT_TARGETS)
def test_clean_interpreter_import(mod, tmp_path):
    """まっさらなインタプリタ（cwd も別の場所）から import できる。テストの副作用に依存していない。"""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", f"import {mod}"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]


@pytest.mark.parametrize("mod", CLEAN_IMPORT_TARGETS)
def test_import_has_no_filesystem_side_effects_in_cwd(mod, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", f"import {mod}"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    assert list(tmp_path.iterdir()) == []


def test_import_does_not_change_cwd():
    before = os.getcwd()
    importlib.import_module("github.run.run_test")
    assert os.getcwd() == before


def test_importing_without_repo_root_on_path_fails_cleanly(tmp_path):
    """`github.` 接頭辞付き import なので、リポジトリのルートが sys.path に無いと import できない（README に無い前提）。"""
    r = subprocess.run([sys.executable, "-c", "import github.run.run_test"], cwd=tmp_path,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "No module named 'github'" in r.stderr


def test_modules_register_gin_configurables():
    import gin

    for n in ("github.run.run_test.run_test",
              "github.simul_bidding_env.Controller.Controller.Controller",
              "github.simul_bidding_env.Environment.BiddingEnv.BiddingEnv",
              "github.simul_bidding_env.Tracker.BiddingTracker.BiddingTracker",
              "github.simul_bidding_env.Tracker.PlayerAnalysis.PlayerAnalysis"):
        assert gin.config._REGISTRY.get_match(n) is not None, n


def test_importing_main_scripts_leaks_strategy_train_env_into_sys_path():
    """本家の副作用: main_*.py を import すると sys.path に strategy_train_env が足され、bidding_train_env が見えるようになる。
    （プレイヤー戦略の選ばれ方が実行順で変わるので、conftest の autouse fixture で毎回取り除いている）"""
    code = ("import sys, github.strategy_train_env.main.main_bc;"
            "print(any(p.endswith('strategy_train_env') for p in sys.path))")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.stdout.strip().splitlines()[-1] == "True", r.stderr[-500:]


MAIN_SCRIPTS = sorted(p.name for p in (GITHUB_DIR / "strategy_train_env" / "main").glob("main_*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", MAIN_SCRIPTS)
def test_training_main_script_starts_without_pythonpath(script, tmp_path):
    """README の `python main/main_xxx.py` が PYTHONPATH なしで起動できる（修正前は No module named 'github'）。
    クローン直後は学習データが無いので、import を通過したあと「データが無い」ことで止まるのが正しい。"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-W", "ignore", str(GITHUB_DIR / "strategy_train_env" / "main" / script)],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert "No module named" not in r.stderr, r.stderr[-800:]
    assert "Traceback" not in r.stderr or "data" in r.stderr  # 止まるなら ./data/traffic/... が無いことが理由
    if script == "main_onlineLp.py":
        assert r.returncode == 0  # OnlineLP は CSV が 0 件でも何もせず正常終了する
    else:
        assert r.returncode != 0 and "data" in r.stderr
