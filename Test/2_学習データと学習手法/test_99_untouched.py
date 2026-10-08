"""99: テストの実行が、github/ と DB/ を汚していないこと（実行前後の git status が同じ）。最後に実行される。"""
import conftest


def test_git_status_of_github_and_db_is_unchanged_by_the_test_run():
    assert conftest._github_status() == conftest.STATUS_AT_START


def test_no_runtime_output_directory_was_created():
    for rel in ("strategy_train_env/data", "strategy_train_env/saved_model", "data", "saved_model"):
        assert not (conftest.GITHUB_DIR / rel).exists()
    assert not (conftest.REPO_ROOT / "data").exists()      # Test/1 の test_16 は、直下に data/ が無いことを前提にしている
