"""02: 原本から作った学習データ（rlData。1 行 = 広告主 1 社 × 1 ティック）が、仕様どおりであること。"""
import ast
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from conftest import PERIODS, RL_DIR, SUMMARY_DIR

pytestmark = pytest.mark.needs_data
COLUMNS = ["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint", "realAllCost",
           "realAllConversion", "timeStepIndex", "state", "action", "reward", "reward_continuous", "done", "next_state"]


@pytest.fixture(scope="module")
def rl():
    p = RL_DIR / "training_data_all-rlData.csv"
    if not p.exists():
        pytest.skip("学習データが無い。research/src/make_train_data.py を実行する")
    df = pd.read_csv(p)
    df["state"] = df["state"].apply(ast.literal_eval)
    df["next_state"] = df["next_state"].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else None)
    return df


def test_per_period_files_exist():
    assert not [n for n in PERIODS if not (RL_DIR / f"period-{n}-rlData.csv").is_file()]


def test_combined_file_matches_recorded_sha1():
    info = json.loads((SUMMARY_DIR / "training_data_all.json").read_text(encoding="utf-8"))
    h = hashlib.sha1((RL_DIR / "training_data_all-rlData.csv").read_bytes()).hexdigest()
    assert info["sha1"] == h and info["periods"] == PERIODS


def test_columns(rl):
    assert list(rl.columns) == COLUMNS


def test_row_count_is_period_advertiser_tick(rl):
    assert len(rl) == len(PERIODS) * 48 * 48
    assert (rl.groupby(["deliveryPeriodIndex", "advertiserNumber"]).size() == 48).all()


def test_periods_are_in_ascending_order(rl):
    assert list(rl.deliveryPeriodIndex.drop_duplicates()) == [float(n) for n in PERIODS]


def test_state_has_16_finite_dimensions(rl):
    s = np.array(rl["state"].tolist(), dtype=float)
    assert s.shape == (len(rl), 16) and np.isfinite(s).all()


def test_time_left_and_budget_left(rl):
    s = np.array(rl["state"].tolist(), dtype=float)
    np.testing.assert_allclose(s[:, 0], (48 - rl.timeStepIndex) / 48)
    assert (s[:, 1] >= 0).all() and (s[:, 1] <= 1 + 1e-9).all()
    assert (s[rl.timeStepIndex.to_numpy() == 0, 1] == 1.0).all()            # 最初のティックは予算が満額


def test_first_tick_has_no_history(rl):
    s = np.array(rl[rl.timeStepIndex == 0]["state"].tolist(), dtype=float)
    assert (s[:, 2:12] == 0).all() and (s[:, 14:16] == 0).all()


def test_volume_features_are_consistent(rl):
    """state[13]=今のティックの PV 数、state[15]=これまでの PV 数の合計。足し上げると 1 日の PV 数（約 50 万）になる。"""
    for _, g in rl[rl.advertiserNumber == 0].groupby("deliveryPeriodIndex"):
        s = np.array(g.sort_values("timeStepIndex")["state"].tolist(), dtype=float)
        np.testing.assert_array_equal(np.cumsum(s[:, 13])[:-1], s[1:, 15])
        assert 450_000 <= s[:, 13].sum() <= 550_000


def test_action_is_a_nonnegative_finite_alpha(rl):
    assert np.isfinite(rl.action).all() and (rl.action >= 0).all()
    assert 20 < rl.action[rl.action > 0].median() < 200       # α は数十〜百数十（シミュレータの 47 社の中央値は約 70）


def test_rewards(rl):
    assert (rl.reward >= 0).all() and (rl.reward == rl.reward.round()).all()     # reward は購入数（整数）
    assert (rl.reward_continuous >= 0).all()
    # 期待購入数（露出した PV の価値の和）と実際の購入数は、全体ではほぼ釣り合う
    assert rl.reward_continuous.sum() == pytest.approx(rl.reward.sum(), rel=0.1)


def test_done_and_next_state(rl):
    assert (rl[rl.timeStepIndex == 47].done == 1).all()
    assert ((rl.done == 1) == rl["next_state"].isna()).all()
    one = rl[(rl.deliveryPeriodIndex == 7) & (rl.advertiserNumber == 3)].sort_values("timeStepIndex")
    for a, b in zip(one.itertuples(), one.iloc[1:].itertuples()):
        if a.done != 1:
            assert a.next_state == b.state


def test_category_heads_start_with_alpha_15_in_periods_7_to_13(rl):
    """period 7〜13 は、各カテゴリの先頭（位置 0, 8, …, 40）の最初のティックの α が 15（シミュレータの PID の初期値と同じ）。
    period 14 以降は、広告主の並び（どの位置にどの手法がいるか）が違い、先頭が 15 とは限らない。"""
    first = rl[(rl.timeStepIndex == 0) & (rl.advertiserNumber % 8 == 0)]
    np.testing.assert_allclose(first[first.deliveryPeriodIndex <= 13].action, 15.0, rtol=1e-6)
    later = first[first.deliveryPeriodIndex >= 14].action
    assert (np.abs(later - 15.0) > 1e-6).mean() > 0.5


def test_nan_bids_in_raw_data_become_zero_actions(rl):
    """原本で入札額が NaN の広告主・ティック（period 9・13・27 の終盤）は、本家の生成コードが 0 に置き換える。NaN は残らない。"""
    assert rl.drop(columns="next_state").notna().all().all()
    for period, adv, tick in ((13, 1, 47), (27, 44, 47)):
        row = rl[(rl.deliveryPeriodIndex == period) & (rl.advertiserNumber == adv) & (rl.timeStepIndex == tick)]
        assert len(row) == 1 and row.action.iloc[0] == 0
