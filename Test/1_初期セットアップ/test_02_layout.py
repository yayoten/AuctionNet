"""02: クローン直後のファイル構成と、同梱の学習済みモデル・データが揃っているか。"""
import pickle
from pathlib import Path

import pandas as pd
import pytest
import torch

from conftest import GITHUB_DIR, SIM_DIR, TRAIN_DIR, REPO_ROOT

OFFICIAL = SIM_DIR / "strategy" / "official_agent"

TOP_LEVEL_FILES = ["main_test.py", "requirements.txt", "README.md", "LICENSE.txt", "config/test.gin",
                   "run/run_test.py", "run/__init__.py", "pre_generated_dataset/readme_dataset.md"]


@pytest.mark.parametrize("rel", TOP_LEVEL_FILES)
def test_top_level_file_exists(rel):
    p = GITHUB_DIR / rel
    assert p.is_file(), rel
    assert p.stat().st_size > 0 or rel.endswith("__init__.py")


@pytest.mark.parametrize("rel", ["assets", "config", "pre_generated_dataset", "results", "run",
                                 "simul_bidding_env", "strategy_train_env"])
def test_top_level_dir_exists(rel):
    assert (GITHUB_DIR / rel).is_dir()


@pytest.mark.parametrize("sub", ["Controller", "Environment", "PvGenerator", "Tracker", "strategy"])
def test_simul_env_subpackages_have_init(sub):
    assert (SIM_DIR / sub / "__init__.py").is_file()


@pytest.mark.parametrize("sub", ["baseline", "common", "offline_eval", "strategy", "train_data_generator"])
def test_train_env_subpackages_have_init(sub):
    assert (TRAIN_DIR / "bidding_train_env" / sub / "__init__.py").is_file()


@pytest.mark.parametrize("sub", ["bc", "bcq", "cql", "iql", "onlineLp", "td3_bc"])
def test_baseline_algorithm_dirs_have_init(sub):
    assert (TRAIN_DIR / "bidding_train_env" / "baseline" / sub / "__init__.py").is_file()


def test_github_is_a_namespace_package_without_init():
    """github/ 自体には __init__.py が無い（namespace package として import される）。"""
    assert not (GITHUB_DIR / "__init__.py").exists()
    import github

    assert list(github.__path__) == [str(GITHUB_DIR)]


# ---------- 学習済みエージェント ----------
JIT_MODELS = {
    "IQL": "IQLtest/iql_model.pth",
    "BC": "BCtest/bc_model.pth",
    "BCQ": "BCQtest/bcq_model.pth",
    "CQL": "CQLtest/cql_model.pth",
    "TD3_BC": "TD3_bctest/td3_bc_model.pth",
    "MOPO": "mbrl_mopo/model/policy_model/best_policy.pt",
    "COMBO": "mbrl_combomicro/model/policy_model/best_policy.pt",
}
NORMALIZE_DICTS = ["IQLtest", "BCtest", "BCQtest", "CQLtest", "TD3_bctest", "mbrl_mopo", "mbrl_combomicro"]
ENV_MODELS = [f"{d}/model/env_model/{n}" for d in ("mbrl_mopo", "mbrl_combomicro")
              for n in ("env_model_0.pt", "env_model_1.pt", "env_model_2.pt", "env_model_3.pt", "elite_env_model.pt")]


@pytest.mark.parametrize("name,rel", JIT_MODELS.items())
def test_official_model_file_exists_and_is_not_empty(name, rel):
    p = OFFICIAL / rel
    assert p.is_file(), f"{name}: {p}"
    assert p.stat().st_size > 1000, f"{name}: ファイルが小さすぎる（Git LFS のポインタ等の疑い）: {p.stat().st_size} bytes"


@pytest.mark.parametrize("name,rel", JIT_MODELS.items())
def test_official_model_is_not_git_lfs_pointer(name, rel):
    head = (OFFICIAL / rel).read_bytes()[:40]
    assert not head.startswith(b"version https://git-lfs"), name


@pytest.mark.parametrize("name,rel", JIT_MODELS.items())
def test_official_model_loads_with_torch_jit(name, rel):
    model = torch.jit.load(str(OFFICIAL / rel))
    assert model is not None


@pytest.mark.parametrize("name,rel", JIT_MODELS.items())
def test_official_model_forward_on_16dim_state(name, rel):
    """戦略は 16 次元の状態を入れて、スカラー(要素1)の alpha を得る前提。"""
    model = torch.jit.load(str(OFFICIAL / rel))
    out = model(torch.zeros(16))
    assert out.numel() == 1, f"{name}: 出力要素数 {out.numel()}"
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("d", NORMALIZE_DICTS)
def test_normalize_dict_pickle_structure(d):
    with open(OFFICIAL / d / "normalize_dict.pkl", "rb") as f:
        nd = pickle.load(f)
    assert isinstance(nd, dict) and nd
    for k, v in nd.items():
        assert isinstance(k, int) and 0 <= k < 16
        assert {"min", "max"} <= set(v)
        assert v["min"] <= v["max"]


@pytest.mark.parametrize("rel", ENV_MODELS)
def test_mbrl_env_model_file_not_empty(rel):
    p = OFFICIAL / rel
    assert p.is_file() and p.stat().st_size > 1000


def test_td3_bc_extra_checkpoint_exists():
    assert (OFFICIAL / "TD3_bctest/td3_bc_model_30000.pth").stat().st_size > 1000


@pytest.mark.parametrize("episode", range(7))
def test_onlinelp_episode_csv(episode):
    df = pd.read_csv(OFFICIAL / "onlineLpTest" / f"episode-{episode}.csv")
    assert {"timeStepIndex", "advertiserCategoryIndex", "cum_cost", "realCPA"} <= set(df.columns)
    assert len(df) > 0
    assert df["timeStepIndex"].between(0, 47).all()
    assert df["advertiserCategoryIndex"].between(0, 5).all()


def test_onlinelp_files_cover_episodes_used_by_controller():
    """Controller は category i (0..5) に OnlineLp(episode=i) と (episode=i+1) を使う → 0..6 が必要。"""
    for ep in range(7):
        assert (OFFICIAL / "onlineLpTest" / f"episode-{ep}.csv").is_file()


# ---------- PV 生成モデル ----------
@pytest.mark.parametrize("rel", [
    "model_utils/check_point/PV_model/Pv_latest.pth",
    "model_utils/check_point/PV_model/Pv_args.pkl",
    "model_utils/check_point/PV_model/Pv_info_dict.pkl",
    "model_utils/check_point/PV_model/Pv_name_dict.pkl",
    "model_utils/data/PV_time_category_relation_data.pkl",
    "model_utils/data/PV_time_distribution_data.pkl",
    "model_utils/data/generated_105K_pv_data.csv",
    "model_utils/stats_useful/str_key_map_dict.pkl",
    "train_generation_module/stats_useful/str_key_map_dict.pkl",
    "train_generation_module/stats_useful/new_postcode_dict.pkl",
    "train_generation_module/stats_useful/all_str_key_map_dict.pkl",
])
def test_pv_generator_resource_exists(rel):
    p = SIM_DIR / "PvGenerator" / rel
    assert p.is_file() and p.stat().st_size > 0


def test_pv_checkpoint_loads():
    ck = torch.load(SIM_DIR / "PvGenerator/model_utils/check_point/PV_model/Pv_latest.pth", map_location="cpu")
    assert ck is not None


def test_pv_pickles_load():
    for rel in ["model_utils/check_point/PV_model/Pv_args.pkl", "model_utils/check_point/PV_model/Pv_name_dict.pkl",
                "model_utils/data/PV_time_distribution_data.pkl"]:
        with open(SIM_DIR / "PvGenerator" / rel, "rb") as f:
            assert pickle.load(f) is not None


def test_generated_pv_csv_is_readable():
    df = pd.read_csv(SIM_DIR / "PvGenerator/model_utils/data/generated_105K_pv_data.csv", nrows=1000)
    assert len(df) == 1000 and df.shape[1] > 1


# ---------- データ／生成物 ----------
def test_pre_generated_dataset_is_readme_only_and_documents_download_urls():
    """クローン直後は README のみ（80GB のデータは別途ダウンロード）。URL が 11 個書かれていること。"""
    text = (GITHUB_DIR / "pre_generated_dataset/readme_dataset.md").read_text(encoding="utf-8")
    assert text.count("autoBidding_general_track_final_data_period_") == 11
    assert "c18" in text and "isEnd" in text


def test_clone_has_no_trained_strategy_artifacts():
    """strategy_train_env/saved_model と data は同梱されていない（学習して作る前提）。"""
    assert not (TRAIN_DIR / "saved_model").exists()
    assert not (TRAIN_DIR / "data").exists()


def test_assets_images_exist():
    assert any(GITHUB_DIR.joinpath("assets").iterdir())


def test_results_dir_is_empty_placeholder():
    assert list((GITHUB_DIR / "results").iterdir()) == []


def test_license_is_apache_2():
    assert "Apache License" in (GITHUB_DIR / "LICENSE.txt").read_text(encoding="utf-8")


def test_python_files_are_all_syntactically_valid():
    import ast

    bad = []
    for f in GITHUB_DIR.rglob("*.py"):
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:  # pragma: no cover
            bad.append((str(f), str(e)))
    assert not bad


def test_every_py_directory_is_a_package_or_script_dir():
    """__init__.py が無い、.py を含むディレクトリの一覧を把握しておく（増減の検知）。"""
    dirs = {f.parent for f in GITHUB_DIR.rglob("*.py")}
    no_init = sorted(str(d.relative_to(GITHUB_DIR)) for d in dirs if not (d / "__init__.py").exists())
    assert "." in no_init  # github/ 直下は namespace package
