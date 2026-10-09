"""99: テスト実行が github/ を汚していないこと（実行前後の git status が同じ）。最後に実行される（ファイル名順）。

github/ 自体は変更してよい。ここで見るのは「テストが勝手にファイルを作った／書き換えた」ことの検出だけ。
"""
import pytest

import conftest
from conftest import GITHUB_DIR


def test_git_status_of_github_is_unchanged_by_the_test_run():
    assert conftest._github_status() == conftest.GITHUB_STATUS_AT_START


@pytest.mark.parametrize("rel", ["strategy_train_env/data", "strategy_train_env/saved_model", "data", "saved_model"])
def test_no_runtime_output_directory_was_created(rel):
    assert not (GITHUB_DIR / rel).exists()


def test_results_dir_still_empty():
    assert not (GITHUB_DIR / "results").exists() or list((GITHUB_DIR / "results").iterdir()) == []


def test_no_pickle_cache_left_next_to_data():
    assert not list(GITHUB_DIR.rglob("raw_data.pickle"))
