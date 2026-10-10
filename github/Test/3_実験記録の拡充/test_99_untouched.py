"""99: テストの実行が、github/ と DB/ を汚していないこと（実行前後の git status が同じ）。最後に実行される（ファイル名順）。"""
import conftest


def test_git_status_of_github_and_db_is_unchanged_by_the_test_run():
    assert conftest._status() == conftest.STATUS_AT_START
